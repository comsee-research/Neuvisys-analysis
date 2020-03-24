#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:17 2020

@author: thomas
"""

from sklearn.model_selection import ParameterSampler
import json
import subprocess
import os
import numpy as np
from PIL import Image
from natsort import natsorted
from scipy.stats import norm

param_grid1 = {"NEURON_WIDTH": 10,
              "NEURON_HEIGHT": 10,
              "NEURON_SYNAPSES": 2,
              "X_ANCHOR_POINT": 70,
              "Y_ANCHOR_POINT": 70,
              "NETWORK_WIDTH": 12,
              "NETWORK_HEIGHT": 18,
              "NETWORK_DEPTH": 4,
              
              "DELTA_VP": 0.06,
              "DELTA_VD": 0.02,
              "TAU_LTP": 10000,
              "TAU_LTD": 20000,
              "VTHRESH": 20,
              "VRESET": -10,
              "TAU_M": 10000,
              "TAU_INHIB": 5000,
              "NORM_FACTOR": 4,
              "NORM_THRESHOLD": 10}

param_grid2 = {"NEURON_WIDTH": 10,
              "NEURON_HEIGHT": 10,
              "NEURON_SYNAPSES": 2,
              "X_ANCHOR_POINT": 70,
              "Y_ANCHOR_POINT": 70,
              "NETWORK_WIDTH": 12,
              "NETWORK_HEIGHT": 18,
              "NETWORK_DEPTH": 4,
              
              "DELTA_VP": 0.088,
              "DELTA_VD": 0.037,
              "TAU_LTP": 5500,
              "TAU_LTD": 12000,
              "VTHRESH": 13.5,
              "VRESET": -20,
              "TAU_M": 9400,
              "TAU_INHIB": 6700,
              "NORM_FACTOR": 4,
              "NORM_THRESHOLD": 10}

param_grid3 = {"NEURON_WIDTH": 10,
              "NEURON_HEIGHT": 10,
              "NEURON_SYNAPSES": 2,
              "X_ANCHOR_POINT": 70,
              "Y_ANCHOR_POINT": 70,
              "NETWORK_WIDTH": 12,
              "NETWORK_HEIGHT": 18,
              "NETWORK_DEPTH": 4,
              
              "DELTA_VP": 0.077,
              "DELTA_VD": 0.011,
              "TAU_LTP": 6600,
              "TAU_LTD": 13900,
              "VTHRESH": 19,
              "VRESET": -20,
              "TAU_M": 17500,
              "TAU_INHIB": 7700,
              "NORM_FACTOR": 4,
              "NORM_THRESHOLD": 10}

sampler = [param_grid1, param_grid2, param_grid3]
directory = "/home/thomas/neuvisys-analysis/results/batch_5/"
n_iter = 3
# sampler = list(ParameterSampler(param_grid, n_iter))

for i in range(n_iter):
    with open(directory+"configs/"+str(i)+".json", "w") as file:
        json.dump(sampler[i], file)
        
    try:
        os.mkdir(directory+"images/"+str(i))
        os.mkdir(directory+"weights/"+str(i))
    except:
        pass
    
    with open("/home/thomas/neuvisys-dv/configs/conf.json", "w") as conf:
        json.dump({"SAVE_DATA": True,
                   "SAVE_DATA_LOCATION": directory+"weights/"+str(i)+"/",
                   "CONF_FILES_LOCATION": directory+"configs/"+str(i)+".json"}, conf)

    try:
        subprocess.run(["dv-runtime", "-b0"], timeout=450)
    except:
        pass
    
    print("Finished job: "+str(i))

    dire = directory+"weights/" + str(i) + "/"
    files = natsorted([f for f in os.listdir(dire) if f.endswith(".npy")])
    weights = [np.moveaxis(np.concatenate((np.load(dire + file), np.zeros((1, sampler[i]["NEURON_WIDTH"], sampler[i]["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    
    for k, img in enumerate(weights):
        img = np.array(255 * (img / img.max()), dtype=np.uint8)
        img = Image.fromarray(img).save(directory+"images/"+str(i)+"/"+str(k)+".png")

    # for j in range(2):
    #     weights = [np.moveaxis(np.concatenate((np.load(directory + file)[:, j], np.zeros((1, sampler[i]["NEURON_WIDTH"], sampler[i]["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    #     for k, img in enumerate(weights):
    #         img = np.array(255 * (img / img.max()), dtype=np.uint8)
    #         img = Image.fromarray(img).save("/home/thomas/neuvisys-analysis/results/images/weights_"+str(i)+"/syn_"+str(j)+"/"+str(k)+".png")
    
# "DELTA_VP": norm(loc=0.065, scale=0.015),
# "DELTA_VD": norm(loc=0.025, scale=0.015),
# "TAU_LTP": norm(loc=10000, scale=5000),
# "TAU_LTD": norm(loc=15000, scale=5000),
# "VTHRESH": norm(loc=17, scale=4.5),
# "VRESET": [-20],
# "TAU_M": norm(loc=12000, scale=6000),
# "TAU_INHIB": norm(loc=8500, scale=3500),