#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_params

directory = "/home/thomas/neuvisys-dv/configuration/network/"

spike_trains = []
for i in range(3346):
    spike_trains.append(np.array(load_params(directory+"weights/neuron_"+str(i)+".json")["spike_train"]))

argmax = np.argmax([len(spike_trains[i]) for i in range(len(spike_trains))])

spike_train = spike_trains[argmax]
isi = (spike_train[1:] - spike_train[:-1]) / 1000

# Plot spike train
plt.figure()
plt.eventplot(spike_train)

# ISI histogram
plt.figure()
plt.hist(isi[isi < 2000], bins="auto")