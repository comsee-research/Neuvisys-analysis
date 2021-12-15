#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import matplotlib.pyplot as plt
import numpy as np


def spike_plot(spike_train, savefig):
    plt.figure(figsize=(40, 8))
    plt.title("Event plot")
    plt.eventplot(spike_train * 1e-6)
    plt.xlabel("time (s)")
    plt.ylabel("neuron")
    if savefig:
        plt.savefig(savefig)
    plt.show()


def spike_plots_inhibited_neurons(spinet, layer, neuron_id):
    plt.figure(figsize=(16, 6))
    plt.xlabel("time (µs)")
    plt.ylabel("neurons")

    indices = spinet.neurons[layer][neuron_id].params["inhibition_connections"]
    colors1 = ["C{}".format(i) for i in range(len(indices) + 1)]

    eveplot = []
    for i in np.sort(indices + [neuron_id]):
        eveplot.append(spinet.spikes[layer][i][spinet.spikes[layer][i] != 0])

    plt.eventplot(eveplot, colors=colors1)


def plot_isi_histogram(spike_train):
    isi = np.diff(spike_train) / 1000
    isi = isi[isi > 0]

    fig = plt.figure(figsize=(16, 6))
    plt.title("Neuron ISI histogram")
    plt.xlabel("interspike interval (ms)")
    plt.hist(isi, bins=np.arange(0, 700, 25))
    return isi, fig


def spike_rate_histogram(layer_spike_train, savefig=None):
    spike_count = np.count_nonzero(layer_spike_train, axis=1)
    plt.figure(figsize=(16, 6))
    plt.title("Spike rate histogram")
    plt.hist(spike_count / (np.max(layer_spike_train) * 1e-6))
    plt.xlabel("spike / s")
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def rate_variation(layer_spike_train, timebin):
    signal = []
    tstep = np.arange(0, np.max(layer_spike_train), timebin)
    for i in range(tstep.size - 1):
        signal.append(np.sum((layer_spike_train > tstep[i]) & (layer_spike_train < tstep[i + 1])))
    return np.array(signal), tstep


def spike_rate_variation(layer_spike_train, timebin=1000000, savefig=None):  # 1sec
    signal, tstep = rate_variation(layer_spike_train, timebin)
    plt.figure(figsize=(40, 8))
    plt.title("Spike rate variation (time bin = " + str(timebin / 1e3) + "ms)")
    plt.plot(tstep[:-1] / 1e6, signal)
    plt.xlabel("time (s)")
    plt.ylabel("number of spikes")
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def spike_stats(spike_trains):
    for layer, layer_spike_train in enumerate(spike_trains):
        print("Layer " + str(layer + 1) + ", nb neurons = " + str(layer_spike_train.shape[0]) + ":")
        spike_rate_histogram(layer_spike_train)
        spike_rate_variation(layer_spike_train, 100000)
