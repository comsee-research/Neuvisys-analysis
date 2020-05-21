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
nb_neurons = 3346

def spike_train(directory, neuron_id):
    return np.array(load_params(directory+"weights/neuron_"+str(neuron_id)+".json")["spike_train"])

def spike_count(spike_trains, neuron_id):
    return load_params(directory+"weights/neuron_"+str(neuron_id)+".json")["count_spike"]

def std_spikes(spike_counts):
    print("average standard deviation for inhibited neurons")
    nb_spikes_inh = [spike_counts[i:i+4] for i in range(0, len(spike_counts), 4)]
    print(np.mean([np.std(a) for a in nb_spikes_inh]))
    return nb_spikes_inh

def spike_rate_histogram(spike_counts, time):
    plt.figure()
    plt.title("histogram of the neurons spiking rate")
    plt.xlabel("spiking rate (spikes/s)")
    plt.hist(np.array(spike_counts) / time)

def isi_histogram(spike_train):
    isi = (spike_train[1:] - spike_train[:-1]) / 1000
    return isi[isi > 0]

spike_trains = []
spike_counts = []
for i in range(nb_neurons):
    spike_trains.append(spike_train(directory, i))
    spike_counts.append(spike_count(directory, i))

std_spikes = std_spikes(spike_counts)
spike_rate_histogram(spike_counts, 720)

n = 20
argmaxs = np.argpartition([len(spike_trains[i]) for i in range(len(spike_trains))], -4)[-n:]

isis = []
for arg in argmaxs:
    spike_train = spike_trains[arg]
    isi = isi_histogram(spike_train)
    isis.append(isi)
    
    plt.figure()
    plt.title("ISI histogram")
    plt.xlabel("interspike interval (ms)")
    plt.xticks(np.arange(0, 701, 50))
    plt.xlim(left=0)
    plt.hist(isi[isi < 700], bins=30)
