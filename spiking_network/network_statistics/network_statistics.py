#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:25 2020

@author: thomas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

from itertools import combinations
from PIL import Image
from natsort import natsorted


def inhibition_boxplot(directory):
    dist = []
    for network_id in range(2):
        direc = directory + "/images/" + str(network_id) + "/"
        images = natsorted(os.listdir(direc))
        distances = []

        for i in range(0, 3536, 4):
            vectors = []
            for j in range(4):
                vectors.append(
                    np.asarray(Image.open(direc + images[i + j]), dtype=float) / 255
                )

            for combination in combinations(vectors, 2):
                distances.append(np.sum((combination[0] - combination[1]) ** 2) / 100)
        dist.append(distances)

    fig, axes = plt.subplots()

    fig.subplots_adjust(left=0.3, right=0.8)

    axes.set_title("", y=-0.1, fontsize=14)
    axes.set_ylabel("Squared Euclidean distance", fontsize=14)
    axes.boxplot([dist[1], dist[0]])
    axes.xaxis.set_ticklabels(["No inhibition", "Inhibition"], fontsize=14)
    plt.savefig("boxplots.pdf", bbox_inches="tight")
    
def update_dataframe(df: list, spinet):
    conf = {}
    for d in (spinet.conf, spinet.simple_conf):
        conf.update(d)
        
    conf["mean_sr"], conf["std_sr"] = network_spike_rate(spinet)
    conf["mean_isi"], conf["std_isi"] = network_isi(spinet)
    conf["mean_thres"], conf["std_thres"] = network_thresholds(spinet)
    df.append(conf)

def network_params(network_path, nb_networks, trim_sim_val=False):
    conf_list = []
    for conf_name in [
        "/configs/network_config.json",
        "/configs/simple_cell_config.json",
        "/configs/complex_cell_config.json",
    ]:
        net_list = []
        for i in range(nb_networks):
            with open(network_path + str(i) + conf_name) as file:
                net_list.append(json.load(file))

        df = pd.DataFrame(net_list)

        if trim_sim_val:
            try:
                nunique = df.apply(pd.Series.nunique)
                cols_to_drop = nunique[nunique == 1].index
                df = df.drop(cols_to_drop, axis=1)
            except:
                pass
        conf_list.append(df)
    return conf_list


def network_spike_rate(spinet):
    time = np.max(spinet.sspikes)
    srates = np.count_nonzero(spinet.sspikes, axis=1) / (time * 1e-6)
    return np.mean(srates), np.std(srates)


def network_isi(spinet):
    isi = np.diff(spinet.sspikes)
    isi = isi[isi > 0]
    return np.mean(isi), np.std(isi)

def network_thresholds(spinet):
    thresholds = []
    for neuron in spinet.simple_cells:
        thresholds.append(neuron.params["threshold"])
    return np.mean(thresholds), np.std(thresholds)

def spike_plots_simple_cells(spinet, neuron_id):
    plt.figure()
    plt.xlabel("time (µs)")
    plt.ylabel("neurons")

    indices = spinet.simple_cells[neuron_id].inhibition_connections
    colors1 = ["C{}".format(i) for i in range(len(indices) + 1)]

    eveplot = []
    for i in np.sort(indices + [neuron_id]):
        eveplot.append(spinet.sspikes[i][spinet.sspikes[i] != 0])

    plt.eventplot(eveplot, colors=colors1)


def spike_plots_complex_cells(spinet, neuron_id):
    plt.figure()
    plt.xlabel("time (µs)")
    plt.ylabel("neurons")

    indices = spinet.complex_cells[neuron_id].inhibition_connections
    colors1 = ["C{}".format(i) for i in range(len(indices) + 1)]

    eveplot = []
    for i in indices + [neuron_id]:
        eveplot.append(spinet.cspikes[i][spinet.cspikes[i] != 0])

    plt.eventplot(eveplot, colors=colors1)


def crosscorrelogram(spinet):
    indices = []
    for i in range(spinet.l1width):
        for j in range(spinet.l1height):
            for k in range(spinet.l1depth):
                indices.append(spinet.layout1[i, j, k])

    test = spinet.sspikes[indices, :]

    bin_size = 100000  # µs
    nb_bins = 19

    ref_id = 10

    strain_ref = test[ref_id]

    for i in range(100):
        strain_tar = test[i]

        a = np.zeros(nb_bins - 1)
        for tmstp in strain_ref[strain_ref != 0]:
            bins = np.linspace(tmstp - bin_size, tmstp + bin_size, nb_bins)
            a += np.histogram(strain_tar, bins)[0]

        plt.figure()
        plt.title(
            "Crosscorrelogram reference neuron "
            + str(ref_id)
            + " against neuron "
            + str(i)
        )
        plt.ylabel("count")
        plt.xlabel("time delay (ms)")
        plt.plot(np.linspace(-bin_size / 1e3, bin_size / 1e3, nb_bins - 1), a)
        plt.savefig("/home/alphat/Desktop/report/crosscorr_inh/" + str(i))


def orientation_norm_length(spike_vector, angles):
    return np.abs(np.sum(spike_vector * np.exp(2j * angles)) / np.sum(spike_vector))


def direction_norm_length(spike_vector, angles):
    return np.abs(np.sum(spike_vector * np.exp(1j * angles)) / np.sum(spike_vector))


def direction_selectivity(spike_vector):
    return (
        np.max(spike_vector) - spike_vector[(np.argmax(spike_vector) + 8) % 16]
    ) / np.max(spike_vector)


def orientation_selectivity(spike_vector):
    return (
        np.max(spike_vector) - spike_vector[(np.argmax(spike_vector) + 4) % 8]
    ) / np.max(spike_vector)


def weight_centroid(weight):
    weight = weight[0] + weight[1]
    weight /= np.max(weight)
    (X, Y) = np.indices((weight.shape[0], weight.shape[1]))
    x_coord = (X * weight).sum() / weight.sum()
    y_coord = (Y * weight).sum() / weight.sum()
    return x_coord, y_coord


def rf_matching(weights):
    residuals = []
    disparity = []
    for weight in weights:
        res_ref = np.inf
        xmax, ymax = 0, 0
        for x in range(weight.shape[3]):
            for y in range(weight.shape[4]):
                res = weight[:, 0, 0] - np.roll(weight[:, 1, 0], (x, y), axis=(1, 2))

                # fig, ax = plt.subplots(2, 3)
                # fig.suptitle(str(np.sum(np.abs(res))))
                # ax[0, 0].imshow(weight[0, 0, 0])
                # ax[0, 1].imshow(np.roll(weight[0, 1, 0], (x, y), axis=(0, 1)))
                # ax[1, 0].imshow(weight[1, 0, 0])
                # ax[1, 1].imshow(np.roll(weight[1, 1, 0], (x, y), axis=(0, 1)))
                # ax[0, 2].imshow(np.abs(res[0]))
                # ax[1, 2].imshow(np.abs(res[1]))

                res = np.sum(res ** 2)
                if res < res_ref:
                    res_ref = res
                    xmax = x
                    ymax = y
        residuals.append(res_ref)
        disparity.append((xmax, ymax))
    return np.array(residuals), np.array(disparity)


def compute_disparity_0(spinet, disparity, residual, min_resi, max_resi):
    cnt = 0
    fig, axes = plt.subplots(len(spinet.conf["L1YAnchor"])-1, len(spinet.conf["L1XAnchor"]), sharex=False, sharey=True, figsize=(14, 8))
    for i in range(len(spinet.conf["L1XAnchor"])):
        for j in range(len(spinet.conf["L1YAnchor"])):
            mask_residual = (
                residual[
                    cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
                ]
                < max_resi
            ) & (
                residual[
                    cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
                ]
                > min_resi
            )
            if j != 2:
                sns.histplot(
                    disparity[
                        cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"],
                        0,
                    ][mask_residual],
                    ax=axes[j, i],
                    bins=[4.46, 5.575, 7.43, 11.15, 22.3, 50],
                    stat="density",
                )
                plt.setp(axes[j, i].get_xticklabels(), fontsize=15)
                plt.setp(axes[j, i].get_yticklabels(), fontsize=15)
                # axes[j, i].set_xticklabels(np.arange(-5.5, 5.5))
                axes[j, i].set_ylabel('Density', fontsize=22)
            cnt += 1
    
    axes[1, 1].set_xlabel('Disparity (pixel)', fontsize=22)
    plt.savefig("/home/thomas/Desktop/images/estimated", bbox_inches="tight")

    # cnt = 0
    # fig, axes = plt.subplots(len(spinet.conf["L1YAnchor"]), len(spinet.conf["L1XAnchor"]), sharex=False, sharey=True, figsize=(16, 12))
    # for i in range(len(spinet.conf["L1XAnchor"])):
    #     for j in range(len(spinet.conf["L1YAnchor"])):
    #         mask_residual = (
    #             residual[
    #                 cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
    #             ]
    #             < max_resi
    #         ) & (
    #             residual[
    #                 cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
    #             ]
    #             > min_resi
    #         )
    #         # if i != 4 and j != 3:
    #         axes[j, i].set_title(
    #             "nb rf:"
    #             + str(np.count_nonzero(mask_residual))
    #             + "\nmean: "
    #             + str(
    #                 np.mean(
    #                     disparity[
    #                         cnt
    #                         * spinet.conf["L1Depth"] : (cnt + 1)
    #                         * spinet.conf["L1Depth"],
    #                         0,
    #                     ][mask_residual]
    #                 )
    #             )
    #         )
    #         axes[j, i].hist(
    #             disparity[
    #                 cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"], 1
    #             ][mask_residual],
    #             np.arange(-5.5, 6.5),
    #             density=True,
    #         )
    #         cnt += 1


def compute_disparity(
    spinet, disparity, theta, error, residual, epsi_theta, epsi_error, epsi_residual
):
    cnt = 0
    fig, axes = plt.subplots(2, 3, sharex=False, sharey=True, figsize=(16, 12))
    for i in range(5):
        for j in range(4):
            mask_residual = (
                residual[cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth]
                < epsi_residual
            )
            mask_theta = (
                theta[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth]
                > np.pi / 2 + epsi_theta
            ) | (
                theta[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth]
                < np.pi / 2 - epsi_theta
            )
            mask_error = (
                error[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error & mask_theta & mask_residual))
                    + "\nmean: "
                    + str(
                        np.mean(
                            disparity[
                                cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth, 0
                            ][mask_theta & mask_error]
                        )
                    )
                )
                axes[j, i].hist(
                    disparity[cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth, 0][
                        mask_theta & mask_error & mask_residual
                    ],
                    np.arange(-5.5, 6.5),
                    density=True,
                )
            cnt += 1

    cnt = 0
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    for i in range(5):
        for j in range(4):
            mask_residual = (
                residual[cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth]
                < epsi_residual
            )
            mask_theta = (
                theta[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth] > epsi_theta
            ) & (
                theta[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth]
                < np.pi - epsi_theta
            )
            mask_error = (
                error[0, cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error & mask_theta & mask_residual))
                    + "\nmean: "
                    + str(
                        np.mean(
                            disparity[
                                cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth, 0
                            ][mask_theta & mask_error]
                        )
                    )
                )
                axes[j, i].hist(
                    disparity[cnt * spinet.l1depth : (cnt + 1) * spinet.l1depth, 1][
                        mask_theta & mask_error & mask_residual
                    ],
                    np.arange(-5.5, 6.5),
                    density=True,
                )
            cnt += 1


def spike_rate_evolution(spinet, neuron_id):
    spikes = spinet.sspikes[neuron_id][spinet.sspikes[neuron_id] != 0]

    if spikes.size:
        hist = np.histogram(
            spikes,
            bins=np.arange(spikes[0], spikes[-1], 100000000),
        )[0]
        hist = hist / 100
        plt.figure()
        plt.plot(hist)

        plt.figure()
        plt.hist(
            spikes,
            bins=np.arange(spikes[0], spikes[-1], 1000000),
            density=True,
            cumulative=True,
        )
