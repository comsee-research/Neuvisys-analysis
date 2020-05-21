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

n = 20
argmaxs = np.argpartition([len(spike_trains[i]) for i in range(len(spike_trains))], -4)[-n:]
nb_spikes = np.array([len(spike_trains[i]) for i in range(len(spike_trains))])

print("average standard deviation for inhibited neurons")
nb_spikes_inh = [nb_spikes[i:i+4] for i in range(0, nb_spikes.size, 4)]
print(np.mean([np.std(a) for a in nb_spikes_inh]))

plt.figure()
plt.title("histogram of the neurons spiking rate")
plt.xlabel("spiking rate (spikes/s)")
plt.hist(nb_spikes / 360)

isis = []
for arg in argmaxs:
    spike_train = spike_trains[arg]
    isi = (spike_train[1:] - spike_train[:-1]) / 1000
    isis.append(isi)
    
    plt.figure()
    plt.title("ISI histogram")
    plt.xlabel("interspike interval (ms)")
    plt.xticks(np.arange(0, 701, 50))
    plt.xlim(left=0)
    plt.hist(isi[isi < 700], bins=30)

print("total number of spikes per layer")
for i in range(4):
    print(np.sum(nb_spikes[i::4]))

# plt.figure()
# plt.title("spike train")
# plt.eventplot(spike_train)