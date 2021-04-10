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


def generate_networks(directory, n_iter):
    network_params = {
        "NbCameras": [1],
        "L1Width": [4],
        "L1Height": [4],
        "L1Depth": [100],
        "L1XAnchor": [[0, 153, 306]],
        "L1YAnchor": [[0, 110, 220]],
        "Neuron1Width": [10],
        "Neuron1Height": [10],
        "Neuron1Synapses": [1],
        "L2Width": [1],
        "L2Height": [1],
        "L2Depth": [16],
        "L2XAnchor": [[0, 4, 8]],
        "L2YAnchor": [[0, 4, 8]],
        "Neuron2Width": [4],
        "Neuron2Height": [4],
        "Neuron2Depth": [16],
        "SharingType": ["full"],
        "SaveData": [True],
    }

    neuron_params = {
        "VTHRESH": [20],
        "VRESET": [-20],
        "TRACKING": ["partial"],
        "TAU_SRA": [100000],
        "TAU_RP": [20000],
        "TAU_M": [18000],
        "TAU_LTP": [7000],
        "TAU_LTD": [14000],
        "TARGET_SPIKE_RATE": [0.75],
        "SYNAPSE_DELAY": [0],
        "STDP_LEARNING": [True],
        "NORM_FACTOR": [4],
        "MIN_THRESH": [5],
        "ETA_LTP": [0.00077],
        "ETA_LTD": [-0.00021],
        "ETA_SRA": [0.6],
        "ETA_TA": [1],
        "ETA_RP": [1],
        "ETA_INH": [25],
        "DECAY_FACTOR": [0],
    }

    pooling_neuron_params = {
        "VTHRESH": [3],
        "VRESET": [-20],
        "TRACKING": ["partial"],
        "TAU_M": [20000],
        "TAU_LTP": [20000],
        "TAU_LTD": [20000],
        "STDP_LEARNING": [True],
        "NORM_FACTOR": [10],
        "ETA_LTP": [0.2],
        "ETA_LTD": [0.2],
        "ETA_INH": [25],
        "ETA_RP": [1],
        "TAU_RP": [20000],
        "DECAY_FACTOR": [0],
    }

    create_directories(
        directory, network_params, neuron_params, pooling_neuron_params, n_iter
    )


def create_directories(
    directory, network_params, neuron_params, pooling_neuron_params, n_iter
):
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


def launch_neuvisys_multi_pass(network_path, event_file, nb_pass):
    for path in execute(
        [
            "/home/alphat/neuvisys-dv/cmake-build-release/neuvisys",
            network_path,
            "multi-pass",
            event_file,
            str(nb_pass),
        ]
    ):
        print(path, end="")
