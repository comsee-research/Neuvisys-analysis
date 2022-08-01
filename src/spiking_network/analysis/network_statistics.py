#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:25 2020

@author: thomas
"""

import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from natsort import natsorted
from sklearn.metrics import r2_score

from src.spiking_network.analysis.spike_train import fast_time_histogram
from src.spiking_network.network.neuvisys import SpikingNetwork
from src.events.Events import Events


def event_vs_network_activity(spinet, event_path, bins=50):
    events = Events(event_path)
    events.shift_timestamps_to_0()
    events.crop(93, 50, 160, 160)

    hist_bin = np.arange(0, events.get_timestamps()[-1], int(1e3 * bins))
    event_variation, _ = np.histogram(events.get_timestamps(), bins=hist_bin)
    event_variation_norm = event_variation / np.max(event_variation)

    activity_variation = fast_time_histogram(spinet.spikes[0])
    activity_variation_norm = activity_variation / np.max(activity_variation)

    plt.figure()
    plt.title("Hisogram of event activity vs network activity variation over time")
    plt.plot(event_variation_norm, label="event variation")
    plt.plot(activity_variation_norm, label="activity variation")
    plt.xlabel("Time (bin of 50ms)")
    plt.ylabel("Normalized count")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Event activity variation against network activity variation")
    plt.xlabel("Event activity variation (number of events)")
    plt.ylabel("Network activity variation (number of spikes)")
    plt.scatter(event_variation, activity_variation)

    xseq = np.linspace(0, event_variation.max(), 1000)
    coeffs = np.polyfit(event_variation, activity_variation, deg=1)
    p = np.poly1d(coeffs)
    r2 = r2_score(activity_variation, p(event_variation))
    plt.plot(xseq, coeffs[1] + coeffs[0] * xseq, color="k", lw=2.5)
    plt.annotate("fitting equation: " + "{:.3f}x + {:.3f}\nR² = {:.2f}".format(coeffs[1], coeffs[0], r2), (600, 100),
                 xycoords="figure points")
    plt.show()


def inhibition_weight_against_orientation(spinet):
    rotation_stimulus = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
    for layer in range(len(spinet.neurons)):
        if layer == 0:
            pass
            # lateral_inhibition_weight_sum(spinet, spinet.orientations[layer], rotation_stimulus)
        elif layer == 1:
            top_down_inhibition_weight_sum(spinet, spinet.orientations[layer], rotation_stimulus)


def inhibition_weight_against_disparity(spinet):
    disparities_stimulus = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    top_down_inhibition_weight_sum(spinet, spinet.disparities, np.array(disparities_stimulus))


def top_down_inhibition_weight_sum(spinet, responses: np.ndarray, stimulus: []):
    weight_sums = []
    for complex_cell in spinet.neurons[1][::spinet.l_shape[1, 2]]:
        simple_cells = complex_cell.params["in_connections"]

        weight = []
        for simple_cell in simple_cells:
            weight.append(spinet.neurons[0][simple_cell].weights_tdi)
        weight_sums.append(np.sum(np.array(weight), axis=0) * np.std(responses[:, complex_cell.id]))
    weight_sums = np.array(weight_sums).flatten()

    index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(responses, stimulus)

    plt.figure()
    x = np.arange(len(weight_sums))
    y = weight_sums
    last_index = 0

    full_sum = []
    for index in change_stimulus_indices:
        plt.bar(x[last_index:index], y[last_index:index])
        full_sum.append(np.sum(y[last_index:index]))
        last_index = index

    plt.xticks(change_stimulus_indices, change_stimulus, rotation=45)
    plt.title("Sum of inhibition weights sorted by preferred complex cell orientation")
    plt.ylabel("Sum of inhibition weights")
    plt.xlabel("Complex cell preferred orientation in degrees (°)")
    plt.show()

    plt.figure()
    plt.bar(np.arange(len(full_sum)), full_sum)
    plt.show()


def lateral_inhibition_weight_sum(spinet: SpikingNetwork, responses: np.ndarray, stimulus: []):
    total = np.zeros(8)
    for simple_cell in spinet.neurons[0]:
        responses_subset = responses[:, simple_cell.params["lateral_dynamic_inhibition"]]
        index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(responses_subset, stimulus)

        plt.figure()
        x = np.arange(len(simple_cell.weights_li))
        y = simple_cell.weights_li  # / np.max(simple_cell.weights_li)
        last_index = 0

        full_sum = []
        for index in change_stimulus_indices:
            plt.bar(x[last_index:index], y[last_index:index])
            full_sum.append(np.sum(y[last_index:index]))
            last_index = index
        total += np.array(full_sum)

        plt.xticks(change_stimulus_indices, change_stimulus, rotation=45)
        plt.title("Normalized sum of inhibition weights sorted by preferred orientation")
        plt.ylabel("Normalized sum of inhibition weights")
        plt.xlabel("Complex cell preferred orientation in degrees (°)")
        plt.show()

        plt.figure()
        plt.bar(np.arange(len(full_sum)), full_sum)
        plt.show()
        break

    plt.figure()
    plt.bar(np.arange(len(total)), total)
    plt.show()


def preffered_stimulus(responses, stimulus):
    pref_stimulus = np.argmax(responses, axis=0)
    index_pref_stimulus = np.argsort(pref_stimulus)
    sorted_pref_stimulus = stimulus[np.sort(pref_stimulus)]
    values_change_stimulus, indices_change_stimulus = np.unique(sorted_pref_stimulus, return_index=True)
    return index_pref_stimulus, values_change_stimulus, indices_change_stimulus


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
    for d in (spinet.conf, spinet.simple_conf, spinet.complex_conf, spinet.critic_conf, spinet.actor_conf):
        conf.update(d)

    # conf["mean_sr"], conf["std_sr"] = network_spike_rate(spinet)
    # conf["mean_isi"], conf["std_isi"] = network_isi(spinet)
    # conf["mean_thres"], conf["std_thres"] = network_thresholds(spinet)
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
    time = np.max(spinet.spikes)
    srates = np.count_nonzero(spinet.spikes, axis=1) / (time * 1e-6)
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


def rf_matching(spinet):
    residuals = []
    disparities = []
    for i, weight in enumerate(spinet.weights[0]):
        disparity, residual = rf_disparity_matching(weight)
        disparity[disparity >= 5] -= 10
        residuals.append(residual)
        disparities.append(disparity)
        if np.any(spinet.shared_id):
            for shared in spinet.shared_id[i]:
                spinet.neurons[0][shared].add_disparity(disparity)
        else:
            spinet.neurons[0][i].add_disparity(disparity)

    return np.array(disparities), np.array(residuals)


def rf_disparity_matching(weight: np.ndarray):
    residual_ref = np.inf
    xmax, ymax = 0, 0
    left_to_right_ratio = np.sum(weight[:, 0]) / (np.sum(weight[:, 0]) + np.sum(weight[:, 1]))
    if left_to_right_ratio > 0.9 or left_to_right_ratio < 0.1:
        return np.array([np.nan, np.nan]), residual_ref

    for x in range(weight.shape[3]):
        for y in range(weight.shape[4]):
            residual = weight[:, 0, 0] - np.roll(weight[:, 1, 0], (x, y), axis=(1, 2))
            residual = np.sum(residual ** 2)
            if residual < residual_ref:
                residual_ref = residual
                xmax = x
                ymax = y
    return np.array([xmax, ymax]), residual_ref


def disparity_histogram(spinet, disparity):
    plt.figure()
    plt.hist(disparity[:, 0], bins=np.arange(-8, 9), align="left")
    plt.title("Histogram of simple cell disparities")
    plt.xlabel("Disparity (px)")
    plt.ylabel("Count")
    plt.savefig(spinet.path + "figures/0/disparity_histogram", bbox_inches="tight")
    plt.show()


def compute_disparity_0(spinet, disparity, residuals, xs, ys, mat):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
    # fig.suptitle("Ground Truth Depth estimation per region", fontsize=30)
    cnt = 0

    for i in range(len(spinet.p_shape[0, 0])):
        for j in range(len(spinet.p_shape[0, 1])):
            mask_residual = (
                                    residuals[cnt * spinet.l_shape[0, 2]: (cnt + 1) * spinet.l_shape[0, 2]]
                                    < 30
                            ) & (
                                    residuals[cnt * spinet.l_shape[0, 2]: (cnt + 1) * spinet.l_shape[0, 2]]
                                    > 0.5
                            )
            if j != 2:
                sns.histplot(
                    disparity[
                    cnt * spinet.l_shape[0, 2]: (cnt + 1) * spinet.l_shape[0, 2],
                    0,
                    ][mask_residual],
                    ax=axes[j, i],
                    bins=[4.46, 5.575, 7.43, 11.15, 22.3, 70],
                    stat="density",
                    color="#2C363F",
                )
            cnt += 1

    # for i in range(len(xs)):
    #     for j in range(len(ys)):
    #         if j != 2:
    #             sns.histplot(mat[i, j][mat[i, j] < 70], ax=axes[j, i], stat="density")

    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=15)
        ax.set_ylabel("Density", fontsize=24)
        ax.set_xticks(np.arange(0, 71, 10))

    axes[1, 1].set_xlabel("Depth (m)", fontsize=24)

    plt.savefig("/home/thomas/Desktop/test", bbox_inches="tight")


def weight_variation(spinet, network_path):
    ws = []
    cs = []
    for i in range(1000):
        weights = []
        confs = []
        if os.path.exists(network_path + "weights/intermediate_" + str(i) + "/0/0.npy"):
            for j in range(spinet.l_shape[0][2]):
                w = np.load(network_path + "weights/intermediate_" + str(i) + "/0/" + str(j) + ".npy")
                # with open(network_path + "weights/intermediate_" + str(i) + "/0/" + str(j) + ".json") as file:
                #     confs.append(json.load(file)["count_spike"])
                weights.append(w)
            ws.append(weights)
            cs.append(confs)
        else:
            break

        # spinet.weights[0] = np.array(weights)
        # display_network([spinet])
        # shutil.copy(network_path + "figures/0/weight_sharing_combined.pdf", "/home/thomas/Bureau/weights/" + str(i) + ".pdf")

        # disparities, residuals = rf_matching(spinet)
        # plt.figure()
        # plt.hist(disparities[:, 0], bins=np.arange(-8, 9), align="left")
        # plt.title("Histogram of simple cell disparities")
        # plt.xlabel("Disparity (px)")
        # plt.ylabel("Count")
        # plt.savefig("/home/thomas/Bureau/disparities/" + str(i), bbox_inches="tight")

    ws = np.array(ws)
    cs = np.array(cs)
    total_sum = np.sum(ws, axis=tuple(np.arange(1, ws.ndim)))
    weight_diff = np.abs(np.diff(ws, axis=0))
    sum_diff = np.sum(weight_diff, axis=tuple(np.arange(1, weight_diff.ndim)))

    plt.figure()
    plt.title("Weight change percentage over time")
    plt.ylabel("Weight change in %")
    plt.xlabel("Time (s)")
    plt.plot(100 * sum_diff / total_sum[:-1])


def compute_disparity(
        spinet, disparity, theta, error, residual, epsi_theta, epsi_error, epsi_residual
):
    cnt = 0
    fig, axes = plt.subplots(2, 3, sharex=False, sharey=True, figsize=(16, 12))
    for i in range(5):
        for j in range(4):
            mask_residual = (
                    residual[cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                    < epsi_residual
            )
            mask_theta = (
                                 theta[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                                 > np.pi / 2 + epsi_theta
                         ) | (
                                 theta[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                                 < np.pi / 2 - epsi_theta
                         )
            mask_error = (
                    error[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error & mask_theta & mask_residual))
                    + "\nmean: "
                    + str(
                        np.mean(
                            disparity[
                            cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth, 0
                            ][mask_theta & mask_error]
                        )
                    )
                )
                axes[j, i].hist(
                    disparity[cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth, 0][
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
                    residual[cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                    < epsi_residual
            )
            mask_theta = (
                                 theta[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth] > epsi_theta
                         ) & (
                                 theta[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                                 < np.pi - epsi_theta
                         )
            mask_error = (
                    error[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error & mask_theta & mask_residual))
                    + "\nmean: "
                    + str(
                        np.mean(
                            disparity[
                            cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth, 0
                            ][mask_theta & mask_error]
                        )
                    )
                )
                axes[j, i].hist(
                    disparity[cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth, 1][
                        mask_theta & mask_error & mask_residual
                        ],
                    np.arange(-5.5, 6.5),
                    density=True,
                )
            cnt += 1
