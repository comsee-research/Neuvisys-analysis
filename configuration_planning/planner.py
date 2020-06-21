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
from scipy.stats.distributions import norm

def generate_multiple_configurations(directory, sampler, network_params, n_iter):
    for i in range(n_iter):
        os.mkdir(directory+"network_"+str(i))
        os.mkdir(directory+"network_"+str(i)+"/weights")
        os.mkdir(directory+"network_"+str(i)+"/configs")
        os.mkdir(directory+"network_"+str(i)+"/figures")
        os.mkdir(directory+"network_"+str(i)+"/images")
        
        with open(directory+"network_"+str(i)+"/configs/network_config.json", "w") as file:
            json.dump(network_params, file)
        
        with open(directory+"network_"+str(i)+"/configs/neuron_config.json", "w") as file:
            json.dump(sampler[i], file)

        with open(directory+"conf.json", "w") as conf:
            json.dump({"SAVE_DATA": True,
                       "WEIGHT_SHARING": True,
                       "SAVE_DATA_LOCATION": directory+"network_"+str(i)+"/weights/",
                       "NETWORK_CONFIG": directory+"network_"+str(i)+"/configs/network_config.json"}, conf)
    
        for path in execute(["dv-runtime", "-b0"]):
            print(path, end="")

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        if "Network reset" in stdout_line:
            print("terminating learning")
            popen.terminate()
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
            
network_params = {
	"Neuron1Config": "/home/thomas/neuvisys-dv/configuration/network/configs/neuron_config.json",
	"Neuron2Config": "/home/thomas/neuvisys-dv/configuration/network/configs/pooling_neuron_config.json",

	"L1Width": 34,
	"L1Height": 26,
	"L1Depth": 100,
	"L1XAnchor": 0,
	"L1YAnchor": 0,
	"Neuron1Width": 10,
	"Neuron1Height": 10,
	"Neuron1Synapses": 1,

	"L2Width": 0,
	"L2Height": 0,
	"Neuron2Width": 3,
	"Neuron2Height": 3
}

neuron_params = {
    "SYNAPSE_DELAY": [0],
	"DELTA_VP": [0.0077],
	"DELTA_VD": [0.0021],
	"DELTA_SR": [0.5, 1, 1.5],
	"DELTA_RP": [0.5, 1, 1.5],
	"DELTA_SRA": [0.03, 0.06, 0.09, 0.12],
	"DELTA_INH": [50, 75, 100],
	"TAU_LTP": [7000],
	"TAU_LTD": [14000],
	"TAU_M": [18000],
	"TAU_RP": [20000, 25000, 30000],
	"TAU_SRA": [50000, 100000, 150000],
	"VTHRESH": [20, 30, 40],
	"VRESET": [-20],
	"DECAY_FACTOR": [0.999, 0.995, 0.99],
	"NORM_FACTOR": [3, 4, 5],
	"TARGET_SPIKE_RATE": [0.75, 1]
}

directory = "/home/thomas/neuvisys-dv/configuration/"
sampler = list(ParameterSampler(neuron_params, 20))
for sample in sampler:
    for key in sample.keys():
        if key != "VRESET" and sample[key] < 0:
            sample[key] = 0
    sample["SYNAPSE_DELAY"] = int(sample["SYNAPSE_DELAY"])
n_iter = len(sampler)

generate_multiple_configurations(directory, sampler, network_params, n_iter)