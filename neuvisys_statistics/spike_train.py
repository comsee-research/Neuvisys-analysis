#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_params

def spike_train(directory, neuron_id):
    return np.array(load_params(directory+"weights/neuron_"+str(neuron_id)+".json")["spike_train"])

def spike_count(directory, neuron_id):
    return load_params(directory+"weights/neuron_"+str(neuron_id)+".json")["count_spike"]

def std_spikes(spike_counts):
    print("average number of spikes standard deviation for inhibited neurons")
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
    return isi[isi >= 0]

directory = "/home/thomas/neuvisys-dv/configuration/Run3/network_5/"
run_time = 3000
run_time = 720
nb_neurons = int(3346/2)

spike_trains = []
spike_counts = []
for i in range(nb_neurons):
    spike_trains.append(spike_train(directory, i))
    spike_counts.append(spike_count(directory, i))

std_spikes = std_spikes(spike_counts)
spike_rate_histogram(spike_counts, run_time)

n = 5
argmaxs = np.argpartition([len(spike_trains[i]) for i in range(len(spike_trains))], -4)[-n:]
for arg in argmaxs:
    plt.figure()
    plt.title("Neuron ISI histogram")
    plt.xlabel("interspike interval (ms)")
    plt.hist(isi_histogram(spike_trains[arg]), bins=np.arange(0, 700, 25))

isis = []
hists = []
for i in range(nb_neurons):
    spike_train = spike_trains[i]
    isi = isi_histogram(spike_train)
    isis += list(isi)
    hists.append(np.histogram(isi[isi < 700], bins=np.arange(0, 700, 25))[0])

plt.figure()
plt.title("Average of all ISI histograms")
plt.xlabel("interspike interval (ms)")
plt.hist(isis, bins=np.arange(0, 700, 25))
