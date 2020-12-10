#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_params

def spike_train(directory):
    return np.array(load_params(directory)["spike_train"])

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

def plot_isi_histogram(directory):
    isi = isi_histogram(spike_train(directory))
    
    fig = plt.figure()
    plt.title("Neuron ISI histogram")
    plt.xlabel("interspike interval (ms)")
    plt.hist(isi, bins=np.arange(0, 700, 25))
    return isi, fig
    
def plot_isi_histograms(directory):
    isis = []
    
    for i in range(nb_neurons):
        fig, isi = plot_isi_histogram(directory+"weights/neuron_"+str(i)+".json")
        plt.savefig(directory+"figures/isi_hist/isi_"+str(i))
        isis += list(isi)
    return isis

####

directory = "/home/thomas/neuvisys-dv/configuration/Run1/network_0/"
run_time = 3000
run_time = 720
nb_neurons = int(3346)

# spike_counts = []
# for i in range(nb_neurons):
#     spike_counts.append(spike_count(directory, i))

# std_spikes = std_spikes(spike_counts)
# spike_rate_histogram(spike_counts, run_time)

isis = plot_isi_histograms(directory)

plt.figure()
plt.title("Average of all ISI histograms")
plt.xlabel("interspike interval (ms)")
plt.hist(isis, bins=np.arange(0, 700, 25))
