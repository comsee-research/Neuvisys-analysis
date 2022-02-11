#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis/src")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis/src")
    home = "/home/thomas/"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


from src.spiking_network.network.neuvisys import SpikingNetwork

network_path = "/home/thomas/Desktop/Experiment/WEIGHTS/network_0/"

# %% Generate Spiking Network

spinet = SpikingNetwork(network_path)


# %%

ws = []
cs = []
for i in range(200):
    weights = []
    confs = []
    for j in range(144):
        w = np.load(spinet.path + "weights/intermediate_" + str(i) + "/0/" + str(j) + ".npy")
        with open(spinet.path + "weights/intermediate_" + str(i) + "/0/" + str(j) + ".json") as file:
            confs.append(json.load(file)["count_spike"])
        weights.append(w)
    ws.append(weights)
    cs.append(confs)
ws = np.array(ws)
cs = np.array(cs)

for i in range(200):
    plt.figure()
    plt.title(cs[i])
    plt.imshow(ws[i, 0, 0, 0, 0])
    plt.show()

# %%

weights = np.sum(ws, axis=(2, 3, 4, 5, 6))
