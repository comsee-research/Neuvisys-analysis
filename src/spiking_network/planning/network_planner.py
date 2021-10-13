#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:17 2020

@author: thomas
"""

from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import os
import subprocess
import numpy as np


def create_networks(exec_path, network_path, n_iter, params):
    for i in range(n_iter):
        n_path = network_path + "/network_" + str(i)
        create_network(exec_path + "/neuvisys-exe", n_path)
        
        conf = open_config_files(n_path + "/configs/")
        
        for neuron_key, neuron_value in params.items():
            sample = list(ParameterSampler(neuron_value, 1))
            for key, value in sample[0].items():
                conf[neuron_key][key] = value
                
            save_config_files(n_path + "/configs/" + neuron_key + ".json", conf[neuron_key])

def open_config_files(config_path):
    conf = {}
    for file in os.listdir(config_path):
        with open(config_path + file, "r") as f:
            conf[file[:-5]] = json.load(f)
    return conf


def save_config_files(config_path, conf):
    with open(config_path, "w") as f:
        json.dump(conf, f)


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
    for path in execute([exec_path, network_path, event_file, str(nb_pass)]):
        print(path, end="")


def launch_neuvisys_ros(exec_path, network_path):
    for path in execute([exec_path, network_path]):
        print(path, end="")


def create_network(exec_path, network_path):
    for path in execute([exec_path, network_path]):
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
