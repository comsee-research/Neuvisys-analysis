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

def generate_multiple_configurations(directory, sampler, n_iter, learning_time):
    for i in range(n_iter):
        os.mkdir(directory+"network_"+str(i))
        os.mkdir(directory+"network_"+str(i)+"/weights")
        os.mkdir(directory+"network_"+str(i)+"/configs")
        os.mkdir(directory+"network_"+str(i)+"/figures")
        os.mkdir(directory+"network_"+str(i)+"/images")
        
        with open(directory+"network_"+str(i)+"/configs/config.json", "w") as file:
            json.dump(sampler[i], file)

        with open(directory+"conf.json", "w") as conf:
            json.dump({"SAVE_DATA": True,
                       "SAVE_DATA_LOCATION": directory+"network_"+str(i)+"/weights/",
                       "CONF_FILES_LOCATION": directory+"network_"+str(i)+"/configs/config.json"}, conf)
    
        try:
            subprocess.run(["dv-runtime", "-b0"], timeout=learning_time)
        except:
            print("Finished learning: " + str(i))

# "DELTA_VP": norm(loc=0.065, scale=0.015),
# "DELTA_VD": norm(loc=0.025, scale=0.015),
# "TAU_LTP": norm(loc=10000, scale=5000),
# "TAU_LTD": norm(loc=15000, scale=5000),
# "VTHRESH": norm(loc=17, scale=4.5),
# "VRESET": [-20],
# "TAU_M": norm(loc=12000, scale=6000),
# "TAU_INHIB": norm(loc=8500, scale=3500),

param_grid = {
	"NEURON_WIDTH": 10,
	"NEURON_HEIGHT": 10,
	"NEURON_SYNAPSES": 1,
	"SYNAPSE_DELAY": 0,
	"X_ANCHOR_POINT": 0,
	"Y_ANCHOR_POINT": 0,
	"NETWORK_WIDTH": 34,
	"NETWORK_HEIGHT": 26,
	"NETWORK_DEPTH": 4,
	"DELTA_VP": 0.077,
	"DELTA_VD": 0.021,
	"DELTA_SR": 1,
	"DELTA_RP": 1,
	"DELTA_SRA": 0.06,
	"DELTA_INH": 100,
	"TAU_LTP": 7000,
	"TAU_LTD": 14000,
	"TAU_M": 18000,
	"TAU_RP": 25000,
	"TAU_SRA": 100000,
	"VTHRESH": 30,
	"VRESET": -20,
	"DECAY_FACTOR": 0.995,
	"NORM_FACTOR": 4,
	"TARGET_SPIKE_RATE": 0.75
}

directory = "/home/thomas/neuvisys-dv/configuration/"
sampler = list(ParameterGrid(param_grid))
n_iter = len(sampler)

generate_multiple_configurations(directory, sampler, n_iter, 360)