#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:17 2020

@author: thomas
"""

from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import subprocess
import os
import numpy as np
from spiking_network.network_planning.network_config import NetworkConfig


def generate_networks(directory, params, n_iter):
    conf = NetworkConfig()
    for key in params:
        conf.random_params[key].update(params[key])
    list_params = generate_list_params(conf.random_params, n_iter, "ParameterSampler")
    create_directories(directory, list_params, n_iter)


def generate_network_files(directory):
    conf = NetworkConfig()
    list_params = generate_list_params(conf.params, 1, "ParameterGrid")
    create_directories(directory, list_params, 1)


def generate_list_params(params, n_iter, choice_type):
    if choice_type == "ParameterGrid":
        list_network_params = list(ParameterGrid(params["network_params"]))
        list_neuron_params = list(ParameterGrid(params["neuron_params"]))
        list_pooling_neuron_params = list(
            ParameterGrid(params["pooling_neuron_params"])
        )
        list_motor_neuron_params = list(ParameterGrid(params["motor_neuron_params"]))
    elif choice_type == "ParameterSampler":
        list_network_params = list(ParameterSampler(params["network_params"], n_iter))
        list_neuron_params = list(ParameterSampler(params["neuron_params"], n_iter))
        list_pooling_neuron_params = list(
            ParameterSampler(params["pooling_neuron_params"], n_iter)
        )
        list_motor_neuron_params = list(
            ParameterSampler(params["motor_neuron_params"], n_iter)
        )

    for dc in list_neuron_params:
        for i, e in enumerate(dc):
            if type(dc[e]) is not str and type(dc[e]) is not bool:
                if dc[e] < 1 and dc[e] > -1:
                    dc[e] = float(np.round(dc[e], 5))
                else:
                    dc[e] = float(np.round(dc[e], 2))

    if len(list_network_params) > n_iter:
        np.shuffle(list_network_params)
    else:
        list_network_params = list_network_params * (
            n_iter // len(list_network_params) + 1
        )

    # if len(list_neuron_params) > n_iter:
    #     np.shuffle(list_neuron_params)
    # else:
    #     list_neuron_params = list_neuron_params * (
    #         n_iter // len(list_neuron_params) + 1
    #     )

    if len(list_pooling_neuron_params) > n_iter:
        np.shuffle(list_pooling_neuron_params)
    else:
        list_pooling_neuron_params = list_pooling_neuron_params * (
            n_iter // len(list_pooling_neuron_params) + 1
        )

    if len(list_motor_neuron_params) > n_iter:
        np.shuffle(list_motor_neuron_params)
    else:
        list_motor_neuron_params = list_motor_neuron_params * (
            n_iter // len(list_motor_neuron_params) + 1
        )

    return (
        list_network_params,
        list_neuron_params,
        list_pooling_neuron_params,
        list_motor_neuron_params,
    )


def create_directories(directory, list_params, n_iter):
    for i in range(n_iter):
        os.mkdir(directory + "network_" + str(i))
        os.mkdir(directory + "network_" + str(i) + "/configs")
        os.mkdir(directory + "network_" + str(i) + "/figures")
        os.mkdir(directory + "network_" + str(i) + "/figures/complex_directions")
        os.mkdir(directory + "network_" + str(i) + "/figures/complex_figures")
        os.mkdir(directory + "network_" + str(i) + "/figures/complex_orientations")
        os.mkdir(
            directory + "network_" + str(i) + "/figures/complex_weights_orientations"
        )
        os.mkdir(directory + "network_" + str(i) + "/figures/motor_figures")
        os.mkdir(directory + "network_" + str(i) + "/figures/simple_figures")
        os.mkdir(directory + "network_" + str(i) + "/gabors")
        os.mkdir(directory + "network_" + str(i) + "/gabors/data")
        os.mkdir(directory + "network_" + str(i) + "/gabors/figures")
        os.mkdir(directory + "network_" + str(i) + "/gabors/hists")
        os.mkdir(directory + "network_" + str(i) + "/images")
        os.mkdir(directory + "network_" + str(i) + "/images/simple_cells")
        os.mkdir(directory + "network_" + str(i) + "/images/complex_cells")
        os.mkdir(directory + "network_" + str(i) + "/weights")
        os.mkdir(directory + "network_" + str(i) + "/weights/simple_cells")
        os.mkdir(directory + "network_" + str(i) + "/weights/complex_cells")
        os.mkdir(directory + "network_" + str(i) + "/weights/motor_cells")

        with open(
            directory + "network_" + str(i) + "/configs/network_config.json", "w"
        ) as file:
            json.dump(list_params[0][i], file)

        with open(
            directory + "network_" + str(i) + "/configs/simple_cell_config.json", "w"
        ) as file:
            json.dump(list_params[1][i], file)

        with open(
            directory + "network_" + str(i) + "/configs/complex_cell_config.json", "w"
        ) as file:
            json.dump(list_params[2][i], file)

        with open(
            directory + "network_" + str(i) + "/configs/motor_cell_config.json", "w"
        ) as file:
            json.dump(list_params[3][i], file)


def toggle_learning(spinet, switch):
    with open(spinet.path + "configs/complex_cell_config.json", "r") as file:
        conf = json.load(file)
    conf["STDP_LEARNING"] = switch
    with open(spinet.path + "configs/complex_cell_config.json", "w") as file:
        json.dump(conf, file)

    with open(spinet.path + "configs/simple_cell_config.json", "r") as file:
        conf = json.load(file)
    conf["STDP_LEARNING"] = switch
    with open(spinet.path + "configs/simple_cell_config.json", "w") as file:
        json.dump(conf, file)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def launch_neuvisys_multi_pass(exec_path, network_path, event_file, nb_pass):
    for path in execute([exec_path, network_path, event_file, str(nb_pass),]):
        print(path, end="")


def divide_visual_field(nbrx, nbry, srx, sry):
    spacing_x = (346 - (srx * nbrx)) / (nbrx - 1)
    spacing_y = (260 - (srx * nbrx)) / (nbrx - 1)

    X = []
    for x in range(nbrx):
        X.append(x * (srx + spacing_x))
    Y = []
    for y in range(nbry):
        Y.append(y * (sry + spacing_y))

    return X, Y
