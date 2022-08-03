#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:20:36 2020

@author: thomas
"""

import os
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


def activity_comparison(spinet1, spinet2, bin_size=50, nb_stimulus=None, distribution=None):
    sts_s1 = spike_trains(spinet1.spikes[0])
    sts_c1 = spike_trains(spinet1.spikes[1])

    sts_s2 = spike_trains(spinet2.spikes[0])
    sts_c2 = spike_trains(spinet2.spikes[1])

    sp_train = spinet1.spikes[0].flatten()
    sp_train = sp_train[sp_train != 0]

    vbars = None
    if nb_stimulus is not None:
        vbars = np.linspace(0, np.max(sp_train), nb_stimulus) / 1e6

    time_histogram_comparison(spinet1, sts_s1, sts_s2, 0, bin_size, distribution, vbars)
    time_histogram_comparison(spinet1, sts_c1, sts_c2, 1, bin_size, distribution, vbars)


def time_histogram_comparison(spinet, sts_control, sts_experiment, layer, bin_size, distribution=None, vbars=None):
    histogram_control = statistics.time_histogram(sts_control, bin_size=bin_size * pq.ms, output='mean')
    histogram_experiment = statistics.time_histogram(sts_experiment, bin_size=bin_size * pq.ms, output='mean')
    units = pq.Quantity(1, 's')
    times = histogram_control.times.rescale(units).magnitude
    width = histogram_control.sampling_period.rescale(units).item()
    histogram_diff = histogram_control.squeeze().magnitude - histogram_experiment.squeeze().magnitude

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 20), sharex=True)
    control_vs_experiment(spinet, ax1, histogram_control, histogram_experiment, times, width, units, layer, vbars)
    if distribution is not None:
        diff_distribution(ax2, histogram_diff, times, width, distribution, vbars)
    plt.show()
    # control_experiment_diff(histogram_diff, times, width, units)


def control_vs_experiment(spinet, ax, histogram_control, histogram_experiment, times, width, units, layer, vbars=None):
    ax.bar(times, histogram_control.squeeze().magnitude, align='edge', width=width, label='Control', color="#5DA9E9")
    ax.bar(times, histogram_experiment.squeeze().magnitude, align='edge', width=width, label='Experiment', alpha=1,
           color="#6D326D")
    ax.set_xlabel(f"Time ({units.dimensionality})")
    ax.set_ylabel("Spike Rate (Hz)")
    ax.set_title("Time histogram function of grating orientation")
    for vbar in vbars:
        ax.axvline(x=vbar, color="#4C3B4D", linestyle='--')
    ax.legend()
    if not os.path.exists(spinet.path + "figures/" + str(layer) + "/activity_comparison"):
        os.mkdir(spinet.path + "figures/" + str(layer) + "/activity_comparison")
    plt.savefig(spinet.path + "figures/" + str(layer) + "/activity_comparison/control_vs_experiment",
                bbox_inches="tight")


def diff_distribution(ax, histogram_diff, times, width, distribution, vbars):
    axbis = ax.twinx()
    ax.set_title("Time histogram difference and learning distribution")
    hist = np.histogram(distribution, bins=np.arange(-9.5, 8.5), density=True)
    axbis.plot(vbars, np.roll(hist[0], 8), color="#A53860", linewidth=4, label="Learning distribution")
    axbis.set_ylabel("Density")

    ax.bar(times, histogram_diff, align='edge', width=width, label="Activity difference", color="#5DA9E9")
    ax.set_xlabel("Input orientation in degree(°)")
    ax.set_ylabel("Spike Rate (Hz)")
    ax.set_xticks(vbars, np.arange(0, 361, 22.5), rotation=45)
    ax.legend(loc="upper left")
    axbis.legend()


def control_experiment_diff(histogram_diff, times, width, units):
    plt.figure()
    plt.bar(times, histogram_diff, align='edge', width=width)
    plt.xlabel(f"Time ({units.dimensionality})")
    plt.ylabel("Spike Rate (Hz)")
    plt.title("Time histogram difference (control - experiment)")
    plt.legend()
    plt.show()


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


def time_histogram(sts, layer, bin_size, path):
    fig, axes = plt.subplots()
    histogram = statistics.time_histogram(sts, bin_size=bin_size * pq.ms, output='rate')
    plot_time_histogram(histogram, units='s', axes=axes)
    axes.set_title("Time histogram")
    if path:
        plt.savefig(path + str(layer) + "/time_histogram", bbox_inches="tight")
    plt.show()


def fast_time_histogram(sp_train, bins=50, display=False):
    sp_train = sp_train.flatten()
    sp_train = sp_train[sp_train != 0]
    sp_train = np.sort(sp_train)

    hist_bin = np.arange(0, sp_train[-1], int(1e3 * bins))
    activity_variation, _ = np.histogram(sp_train, bins=hist_bin)

    if display:
        plt.figure()
        plt.title("Hisogram of network activity variation over time")
        plt.plot(activity_variation, label="activity variation")
        plt.xlabel("Time (bin of " + str(bins) + "ms)")
        plt.ylabel("Normalized count")
        plt.legend()
        plt.show()

    return activity_variation


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


def correlation_coeficient_matrix(sts, layer, bin_size, path):
    fig, axes = plt.subplots()
    binned_spiketrains = BinnedSpikeTrain(sts, bin_size=bin_size * pq.ms)
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
