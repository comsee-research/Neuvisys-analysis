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

param_grid = {"NEURON_WIDTH": [10],
              "NEURON_HEIGHT": [10],
              "NEURON_SYNAPSES": [2],
              "X_ANCHOR_POINT": [70],
              "Y_ANCHOR_POINT": [70],
              "NETWORK_WIDTH": [12],
              "NETWORK_HEIGHT": [18],
              "NETWORK_DEPTH": [4],
              
              "DELTA_VP": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
              "DELTA_VD": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
              "TAU_LTP": [5000, 10000, 15000, 20000],
              "TAU_LTD": [5000, 10000, 15000, 20000],
              "VTHRESH": [10, 20, 30],
              "VRESET": [-10, -20, -30],
              "TAU_M": [5000, 10000, 15000, 20000],
              "TAU_INHIB": [5000, 10000, 15000, 20000],
              "NORM_FACTOR": [4],
              "NORM_THRESHOLD": [10]}

n_iter = 100
sampler = list(ParameterSampler(param_grid, n_iter))
for i in range(n_iter):
    with open("config_files/conf_"+str(i)+".json", "w") as file:
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
                   "CONF_FILES_LOCATION": "/home/thomas/neuvisys-analysis/config_files/conf_"+str(i)+".json"}, conf)

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
    