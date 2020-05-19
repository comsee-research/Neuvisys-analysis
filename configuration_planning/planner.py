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
from natsort import natsorted

# "DELTA_VP": norm(loc=0.065, scale=0.015),
# "DELTA_VD": norm(loc=0.025, scale=0.015),
# "TAU_LTP": norm(loc=10000, scale=5000),
# "TAU_LTD": norm(loc=15000, scale=5000),
# "VTHRESH": norm(loc=17, scale=4.5),
# "VRESET": [-20],
# "TAU_M": norm(loc=12000, scale=6000),
# "TAU_INHIB": norm(loc=8500, scale=3500),


def generate_multiple_configurations(directory, sampler, n_iter):
    for i in range(n_iter):
        with open(directory+"configs/"+str(i)+".json", "w") as file:
            json.dump(sampler[i], file)

        os.mkdir(directory+"weights/"+str(i))
        with open("/home/thomas/neuvisys-dv/configs/conf.json", "w") as conf:
            json.dump({"SAVE_DATA": True,
                       "SAVE_DATA_LOCATION": directory+"weights/"+str(i)+"/",
                       "CONF_FILES_LOCATION": directory+"configs/"+str(i)+".json"}, conf)
    
        try:
            subprocess.run(["dv-runtime", "-b0"], timeout=190)
        except:
            print("Finished learning: " + str(i))


param_grid = {"NEURON_WIDTH": [10], "NEURON_HEIGHT": [10], "NEURON_SYNAPSES": [2, 3], "SYNAPSE_DELAY": [5000, 10000, 15000, 20000, 25000], "X_ANCHOR_POINT": [0], "Y_ANCHOR_POINT": [0], "NETWORK_WIDTH": [34], "NETWORK_HEIGHT": [26], "NETWORK_DEPTH": [1], "DELTA_VP": [0.06], "DELTA_VD": [0.02], "DELTA_SR": [0.1], "TAU_LTP": [10000], "TAU_LTD": [20000], "VTHRESH": [20, 40, 60, 80], "VRESET": [-10], "TAU_M": [10000], "TAU_INHIB": [5000], "NORM_FACTOR": [4], "NORM_THRESHOLD": [4], "TARGET_SPIKE_RATE": [0.5]}
    
directory = "/home/thomas/neuvisys-dv/configuration/"
network_id = 1
sampler = list(ParameterGrid(param_grid))
n_iter = len(sampler)
