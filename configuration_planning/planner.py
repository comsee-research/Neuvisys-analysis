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
                       "WEIGHT_SHARING": False,
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
	"NEURON_WIDTH": [10],
	"NEURON_HEIGHT": [10],
	"NEURON_SYNAPSES": [1],
	"SYNAPSE_DELAY": [0],#norm(loc=10000, scale=5000),
	"X_ANCHOR_POINT": [0],
	"Y_ANCHOR_POINT": [0],
	"NETWORK_WIDTH": [34],
	"NETWORK_HEIGHT": [26],
	"NETWORK_DEPTH": [4],
	"DELTA_VP": [0.077],
	"DELTA_VD": [0.021],
	"DELTA_SR": norm(loc=1, scale=0.5),
	"DELTA_RP": norm(loc=1, scale=0.5),
	"DELTA_SRA": norm(0.06, scale=0.1),
	"DELTA_INH": norm(100, scale=50),
	"TAU_LTP": [7000],
	"TAU_LTD": [14000],
	"TAU_M": norm(18000, scale=5000),
	"TAU_RP": norm(25000, scale=10000),
	"TAU_SRA": norm(100000, scale=50000),
	"VTHRESH": [30],
	"VRESET": [-20],
	"DECAY_FACTOR": [0.999, 0.995, 0.99],
	"NORM_FACTOR": [4],
	"TARGET_SPIKE_RATE": norm(0.75, scale=0.35)
}

directory = "/home/thomas/neuvisys-dv/configuration/"
sampler = list(ParameterSampler(param_grid, 20))
for sample in sampler:
    for key in sample.keys():
        if key != "VRESET" and sample[key] < 0:
            sample[key] = 0
    sample["SYNAPSE_DELAY"] = int(sample["SYNAPSE_DELAY"])
n_iter = len(sampler)

generate_multiple_configurations(directory, sampler, n_iter, 3400)