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

param_grid = {"NEURON_WIDTH": [10],
              "NEURON_HEIGHT": [10],
              "NEURON_SYNAPSES": [2],
              "X_ANCHOR_POINT": [70],
              "Y_ANCHOR_POINT": [70],
              "NETWORK_WIDTH": [12],
              "NETWORK_HEIGHT": [18],
              "NETWORK_DEPTH": [4],
              
              "DELTA_VP": norm(loc=0.065, scale=0.015),
              "DELTA_VD": norm(loc=0.025, scale=0.015),
              "TAU_LTP": norm(loc=10000, scale=5000),
              "TAU_LTD": norm(loc=15000, scale=5000),
              "VTHRESH": norm(loc=17, scale=4.5),
              "VRESET": [-20],
              "TAU_M": norm(loc=12000, scale=6000),
              "TAU_INHIB": norm(loc=8500, scale=3500),
              "NORM_FACTOR": [4],
              "NORM_THRESHOLD": [10]}

n_iter = 100
sampler = list(ParameterSampler(param_grid, n_iter))
for i in range(50, n_iter+50):
    with open("results/config_files/conf_"+str(i)+".json", "w") as file:
        json.dump(sampler[i], file)
        
    try:
        os.mkdir("results/metrics/"+str(i))
        os.mkdir("results/weights/"+str(i))
        # os.mkdir("results/metric/weights_"+str(i)+"/syn_0")
        # os.mkdir("results/metric/weights_"+str(i)+"/syn_1")
    except:
        pass
    
    with open("/home/thomas/neuvisys-dv/configs/conf.json", "w") as conf:
        json.dump({"SAVE_DATA": True,
                   "SAVE_DATA_LOCATION": "/home/thomas/neuvisys-analysis/results/weights/"+str(i)+"/",
                   "CONF_FILES_LOCATION": "/home/thomas/neuvisys-analysis/results/config_files/conf_"+str(i)+".json"}, conf)

    try:
        subprocess.run(["dv-runtime", "-b0"], timeout=400)
    except:
        pass
    
    print("Finished job:"+str(i))

    directory = "/home/thomas/neuvisys-analysis/results/weights/" + str(i) + "/"
    files = natsorted([f for f in os.listdir(directory) if f.endswith(".npy")])
    weights = [np.moveaxis(np.concatenate((np.load(directory + file), np.zeros((1, sampler[i]["NEURON_WIDTH"], sampler[i]["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    
    for k, img in enumerate(weights):
        img = np.array(255 * (img / img.max()), dtype=np.uint8)
        img = Image.fromarray(img).save("/home/thomas/neuvisys-analysis/results/metrics/"+str(i)+"/"+str(k)+".png")

    # for j in range(2):
    #     weights = [np.moveaxis(np.concatenate((np.load(directory + file)[:, j], np.zeros((1, sampler[i]["NEURON_WIDTH"], sampler[i]["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    #     for k, img in enumerate(weights):
    #         img = np.array(255 * (img / img.max()), dtype=np.uint8)
    #         img = Image.fromarray(img).save("/home/thomas/neuvisys-analysis/results/metric/weights_"+str(i)+"/syn_"+str(j)+"/"+str(k)+".png")
    