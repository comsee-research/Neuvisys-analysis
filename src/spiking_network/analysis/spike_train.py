#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import matplotlib.pyplot as plt
import numpy as np

from neo import SpikeTrain
import quantities as pq
from viziphant.rasterplot import eventplot, rasterplot_rates
from viziphant.statistics import plot_isi_histogram, plot_instantaneous_rates_colormesh, plot_time_histogram
from elephant import statistics, kernels
from viziphant.spike_train_correlation import plot_corrcoef
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient

from scipy.stats import laplace


def time_histogram_comparison(sts_control, sts_experiment):
    plt.figure()
    histogram_control = statistics.time_histogram(sts_control, bin_size=100 * pq.ms, output='mean')
    histogram_experiment = statistics.time_histogram(sts_experiment, bin_size=100 * pq.ms, output='mean')

    units = pq.Quantity(1, 's')
    width = histogram_control.sampling_period.rescale(units).item()
    times = histogram_control.times.rescale(units).magnitude

    # Create the plot
    # plt.bar(times, histogram_control.squeeze().magnitude, align='edge', width=width, label='control')
    # plt.bar(times, histogram_experiment.squeeze().magnitude, align='edge', width=width, label='experiment', alpha=0.6)
    # plt.xlabel(f"Time ({units.dimensionality})")
    # plt.ylabel("Spike Rate (Hz)")
    #
    # plt.title("Time histogram")
    # plt.legend()
    # plt.show()

    plt.figure()
    histogram_diff = histogram_control.squeeze().magnitude - histogram_experiment.squeeze().magnitude
    plt.bar(times, histogram_diff, align='edge', width=width)

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axes = gs.subplots(sharex=True)
    axes[0].bar(11.25*times[:len(times)//2], histogram_diff[:len(histogram_diff)//2])

    rotations = [0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338]
    draw = laplace(0, 1).rvs(size=50)
    draw = np.round(draw)

    hist = np.histogram(draw, bins=np.arange(-8.5, 8.5), density=True)
    axes[1].bar(rotations, np.roll(hist[0], 8), width=22.5)


def spike_trains(strains: np.array):
    sts = []
    tstop = np.max(strains)
    for spike_train in strains:
        sts.append(SpikeTrain(spike_train[spike_train > 0], tstop, units=pq.us))
    return sts


def raster_plot(sts, layer, path):
    fig, axes = plt.subplots()
    rasterplot_rates(sts, pophistbins=250, histscale=0.1, ax=axes, markerargs={"marker": '.', "markersize": 1})
    axes.set_title("Raster plot")
    if path:
        plt.savefig(path + str(layer) + "/rasterplot_rates", bbox_inches="tight")
    plt.show()


def event_plot(sts, layer, path):
    fig, axes = plt.subplots()
    eventplot(sts, axes=axes)
    axes.set_ylabel('Neurons')
    axes.set_title("Eventplot")
    if path:
        plt.savefig(path + str(layer) + "/eventplot", bbox_inches="tight")
    plt.show()


<<<<<<< HEAD
def time_histogram(sts, layer, path, axes=None):
    if axes is None:
        fig, axes = plt.subplots()
    histogram = statistics.time_histogram(sts, bin_size=100 * pq.ms, output="rate")
=======
def time_histogram(sts, layer, path):
    fig, axes = plt.subplots()
    histogram = statistics.time_histogram(sts, bin_size=100 * pq.ms, output='rate')
>>>>>>> 6dde7b61e26fce2022a2d039d907bfce0620cf68
    plot_time_histogram(histogram, units='s', axes=axes)
    axes.set_title("Time histogram")
    if path:
        plt.savefig(path + str(layer) + "/time_histogram", bbox_inches="tight")
    # plt.show()


def spike_rate_histogram(sps, layer, path):
    fig, axes = plt.subplots()
    spike_count = np.count_nonzero(sps, axis=1)
    axes.hist(spike_count / (np.max(sps) * 1e-6))
    axes.set_title("Spike rate histogram")
    axes.set_xlabel("spike / s")
    if path:
        plt.savefig(path + str(layer) + "/spike_rates", bbox_inches="tight")
    plt.show()


def isi_histogram(sts, layer, path):
    fig, axes = plt.subplots()
    plot_isi_histogram(sts[0:112], cutoff=1 * pq.s, histtype='bar', axes=axes)
    axes.set_title("ISI histogram")
    if path:
        plt.savefig(path + str(layer) + "/isi_histogram", bbox_inches="tight")
    plt.show()


def instantaneous_rates(sts, layer, path):
    fig, axes = plt.subplots()
    kernel = kernels.GaussianKernel(sigma=100 * pq.ms)
    rates = statistics.instantaneous_rate(sts, sampling_period=100 * pq.ms, kernel=kernel)
    plot_instantaneous_rates_colormesh(rates, axes=axes)
    axes.set_title("Instantaneous rates")
    if path:
        plt.savefig(path + str(layer) + "/instantaneous_rates", bbox_inches="tight")
    plt.show()


def correlation_coeficient_matrix(sts, layer, path):
    fig, axes = plt.subplots()
    binned_spiketrains = BinnedSpikeTrain(sts, bin_size=100 * pq.ms)
    corrcoef_matrix = correlation_coefficient(binned_spiketrains)
    plot_corrcoef(corrcoef_matrix, axes=axes)
    axes.set_xlabel('Neuron')
    axes.set_ylabel('Neuron')
    axes.set_title("Correlation coefficient matrix")
    if path:
        plt.savefig(path + str(layer) + "/corrcoef", bbox_inches="tight")
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
