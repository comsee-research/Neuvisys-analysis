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


def launch_spinet(directory, n_iter):
    network_params = {
        "NbCameras": [1],
        "L1Width": [4],
        "L1Height": [4],
        "L1Depth": [144],
        "L1XAnchor": [[0, 153, 306]],
        "L1YAnchor": [[0, 110, 220]],
        "Neuron1Width": [10],
        "Neuron1Height": [10],
        "Neuron1Synapses": [1],
        "L2Width": [1],
        "L2Height": [1],
        "L2Depth": [16, 24],
        "L2XAnchor": [[0, 4, 8]],
        "L2YAnchor": [[0, 4, 8]],
        "Neuron2Width": [4],
        "Neuron2Height": [4],
        "Neuron2Depth": [100],
        "WeightSharing": [True],
        "SaveData": [True],
        "NetworkPath": [
            "/media/alphat/SSD Games/Thesis/configuration/network/configuration/network/"
        ],
        "Display": [False],
    }

    neuron_params = {
        "SYNAPSE_DELAY": [0],
        "DELTA_VP": [0.00077],
        "DELTA_VD": [0.00021],
        "DELTA_SR": [1],
        "DELTA_RP": [1],
        "DELTA_SRA": [0.06],
        "DELTA_INH": [25],
        "TAU_LTP": [7000],
        "TAU_LTD": [14000],
        "TAU_M": [18000],
        "TAU_RP": [20000],
        "TAU_SRA": [100000],
        "VTHRESH": [20],
        "VRESET": [-20],
        "DECAY_FACTOR": [0],
        "MIN_THRESH": [5],
        "NORM_FACTOR": [4],
        "TARGET_SPIKE_RATE": [0.75],
        "STDP_LEARNING": [True],
        "TRACKING": [False],
    }

    pooling_neuron_params = {
        "VTHRESH": [1, 2, 3, 4],
        "VRESET": [-20],
        "TAU_M": [20000],
        "TAU_LTP": [20000],
        "NORM_FACTOR": [6, 8, 10, 12],
        "DELTA_VP": [0.1, 0.2, 0.3],
        "DELTA_INH": [25],
        "DECAY_FACTOR": [0],
        "STDP_LEARNING": [True],
        "TRACKING": [False],
    }

    generate_networks(
        directory, network_params, neuron_params, pooling_neuron_params, n_iter
    )


def generate_networks(
    directory, network_params, neuron_params, pooling_neuron_params, n_iter
):
    for i in range(n_iter):
        os.mkdir(directory + "network_" + str(i))
        os.mkdir(directory + "network_" + str(i) + "/weights")
        os.mkdir(directory + "network_" + str(i) + "/weights/simple_cells")
        os.mkdir(directory + "network_" + str(i) + "/weights/complex_cells")
        os.mkdir(directory + "network_" + str(i) + "/gabors")
        os.mkdir(directory + "network_" + str(i) + "/gabors/data")
        os.mkdir(directory + "network_" + str(i) + "/gabors/figures")
        os.mkdir(directory + "network_" + str(i) + "/gabors/hists")
        os.mkdir(directory + "network_" + str(i) + "/configs")
        os.mkdir(directory + "network_" + str(i) + "/figures")
        os.mkdir(directory + "network_" + str(i) + "/figures/complex_figures")
        os.mkdir(directory + "network_" + str(i) + "/figures/complex_orientations")
        os.mkdir(directory + "network_" + str(i) + "/images")
        os.mkdir(directory + "network_" + str(i) + "/images/simple_cells")
        os.mkdir(directory + "network_" + str(i) + "/images/complex_cells")

        network_params["Neuron1Config"] = [
            directory + "network_" + str(i) + "/configs/simple_cell_config.json"
        ]
        network_params["Neuron2Config"] = [
            directory + "network_" + str(i) + "/configs/complex_cell_config.json"
        ]
        network_params["SaveDataLocation"] = [directory + "network_" + str(i) + "/"]
        with open(
            directory + "network_" + str(i) + "/configs/network_config.json", "w"
        ) as file:
            json.dump(list(ParameterSampler(network_params, 1))[0], file)

        with open(
            directory + "network_" + str(i) + "/configs/simple_cell_config.json", "w"
        ) as file:
            json.dump(list(ParameterSampler(neuron_params, 1))[0], file)

        with open(
            directory + "network_" + str(i) + "/configs/complex_cell_config.json", "w"
        ) as file:
            json.dump(list(ParameterSampler(pooling_neuron_params, 1))[0], file)


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


def launch_neuvisys_rotation(network_path, event_file, rotation):
    for path in execute(
        [
            "/home/alphat/neuvisys-dv/cmake-build-release/event-analysis",
            network_path,
            "rotation",
            event_file,
            str(rotation),
        ]
    ):
        print(path, end="")


def launch_neuvisys_multi_pass(network_path, event_file, nb_pass):
    for path in execute(
        [
            "/home/alphat/neuvisys-dv/cmake-build-release/event-analysis",
            network_path,
            "multi-pass",
            event_file,
            str(nb_pass),
        ]
    ):
        print(path, end="")


def launch_neuvisys_stereo(network_path, left_file, right_file, nb_pass):
    for path in execute(
        [
            "/home/alphat/neuvisys-dv/cmake-build-release/event-analysis",
            network_path,
            "stereo",
            left_file,
            right_file,
            str(nb_pass),
        ]
    ):
        print(path, end="")
