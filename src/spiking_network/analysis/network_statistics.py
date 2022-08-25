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
from scipy import spatial

import copy

from src.spiking_network.network.neuvisys import SpikingNetwork
import matplotlib

from src.spiking_network.analysis.modif_inhib_visual import (
    visualize_total_inhibition_evolution2,
    visualize_total_tdinhibition_evolution2,
    data_analysis_inhibition
)

def inhibition_weight_against_orientation(spinet):
    rotation_stimulus = np.array(
        [0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
    for layer in range(len(spinet.neurons)):
        if layer == 0:
            pass
            # lateral_inhibition_weight_sum(spinet, spinet.orientations[layer], rotation_stimulus)
        elif layer == 1:
            top_down_inhibition_weight_sum(
                spinet, spinet.orientations[layer], rotation_stimulus)


def inhibition_weight_against_disparity(spinet):
    disparities_stimulus = [-8, -7, -6, -5, -
                            4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    top_down_inhibition_weight_sum(
        spinet, spinet.disparities, np.array(disparities_stimulus))


def top_down_inhibition_weight_sum(spinet, responses: np.ndarray, stimulus: []):
    weight_sums = []
    for complex_cell in spinet.neurons[1][::spinet.l_shape[1, 2]]:
        simple_cells = complex_cell.params["in_connections"]

        weight = []
        for simple_cell in simple_cells:
            weight.append(spinet.neurons[0][simple_cell].weights_tdi)
        weight_sums.append(np.sum(np.array(weight), axis=0)
                           * np.std(responses[:, complex_cell.id]))
    weight_sums = np.array(weight_sums).flatten()

    index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(
        responses, stimulus)

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
    plt.title(
        "Sum of inhibition weights sorted by preferred complex cell orientation")
    plt.ylabel("Sum of inhibition weights")
    plt.xlabel("Complex cell preferred orientation in degrees (°)")
    plt.show()

    plt.figure()
    plt.bar(np.arange(len(full_sum)), full_sum)
    plt.show()


def lateral_inhibition_weight_sum(spinet: SpikingNetwork, responses: np.ndarray, stimulus: []):
    total = np.zeros(8)
    for simple_cell in spinet.neurons[0]:
        responses_subset = responses[:,
                                     simple_cell.params["lateral_dynamic_inhibition"]]
        index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(
            responses_subset, stimulus)

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
        plt.title(
            "Normalized sum of inhibition weights sorted by preferred orientation")
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
    values_change_stimulus, indices_change_stimulus = np.unique(
        sorted_pref_stimulus, return_index=True)
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
                    np.asarray(Image.open(
                        direc + images[i + j]), dtype=float) / 255
                )

            for combination in combinations(vectors, 2):
                distances.append(
                    np.sum((combination[0] - combination[1]) ** 2) / 100)
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
    left_to_right_ratio = np.sum(
        weight[:, 0]) / (np.sum(weight[:, 0]) + np.sum(weight[:, 1]))
    if left_to_right_ratio > 0.9 or left_to_right_ratio < 0.1:
        return np.array([np.nan, np.nan]), residual_ref

    for x in range(weight.shape[3]):
        for y in range(weight.shape[4]):
            residual = weight[:, 0, 0] - \
                np.roll(weight[:, 1, 0], (x, y), axis=(1, 2))
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
    plt.savefig(spinet.path + "figures/0/disparity_histogram",
                bbox_inches="tight")
    plt.show()


def compute_disparity_0(spinet, disparity, residuals, xs, ys, mat):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
    # fig.suptitle("Ground Truth Depth estimation per region", fontsize=30)
    cnt = 0

    for i in range(len(spinet.p_shape[0, 0])):
        for j in range(len(spinet.p_shape[0, 1])):
            mask_residual = (
                residuals[cnt * spinet.l_shape[0, 2]
                    : (cnt + 1) * spinet.l_shape[0, 2]]
                < 30
            ) & (
                residuals[cnt * spinet.l_shape[0, 2]
                    : (cnt + 1) * spinet.l_shape[0, 2]]
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
                error[0, cnt *
                      spinet.l1depth: (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error &
                          mask_theta & mask_residual))
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
                theta[0, cnt *
                      spinet.l1depth: (cnt + 1) * spinet.l1depth] > epsi_theta
            ) & (
                theta[0, cnt * spinet.l1depth: (cnt + 1) * spinet.l1depth]
                < np.pi - epsi_theta
            )
            mask_error = (
                error[0, cnt *
                      spinet.l1depth: (cnt + 1) * spinet.l1depth] < epsi_error
            )
            if i != 4 and j != 3:
                axes[j, i].set_title(
                    "nb rf:"
                    + str(np.count_nonzero(mask_error &
                          mask_theta & mask_residual))
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


def visualize_potentials(spinet: SpikingNetwork, layer_id, neuron_z, visualize=True):
    number_of_displays = len(spinet.stats)
    x = [[]]
    y = [[]]
    step = 0.1
    val_max = 500
    excit_x = [[]]
    excit_y = [[]]
    x_rest = [[]]
    y_rest = [[]]
    x_spike = [[]]
    y_spike = [[]]
    thresh = [[]]
    for i in range(number_of_displays):
        for count, potential_time in enumerate(spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]):
            x[i].append(potential_time[1])
            y[i].append(potential_time[0])
            if(layer_id == 0):
                thresh[i].append(30)
            else:
                thresh[i].append(3)
            if(y[i][-1] > thresh[i][-1]):
                x_rest[i].append(potential_time[1])
                y_rest[i].append(-20)
                x_spike[i].append(x[i][-1])
                y_spike[i].append(y[i][-1])
            if(count+1 < len(spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"])):
                if(x[i][-1] < spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"][count+1][1]):
                    number = int((spinet.stats[i][str(i)][layer_id][str(
                        layer_id)][neuron_z][1]["potential_train"][count+1][1] - x[i][-1]-step) / step)
                    if(number > val_max):
                        number = val_max
                    temp = np.linspace(x[i][-1]+step, spinet.stats[i][str(i)][layer_id][str(
                        layer_id)][neuron_z][1]["potential_train"][count+1][1], number)
                    potential = y[i][-1]
                    time = x[i][-1]
                    for elem in temp:
                        x[i].append(elem)
                        thresh[i].append(thresh[i][-1])
                        if(layer_id == 0):
                            newpotential = potential * \
                                np.exp(-(elem - time) /
                                       (spinet.simple_conf['TAU_M']*1000))
                        else:
                            newpotential = potential * \
                                np.exp(-(elem - time) /
                                       (spinet.complex_conf['TAU_M']*1000))
                        y[i].append(newpotential)
        if(visualize):
            plt.figure(i+1)
            plt.plot(x[i], y[i], 'b-', label='potential of the neuron')

            plt.plot(x[i], thresh[i], 'r--', label='threshold')
            if(len(x[i]) != 0):
                zxrest = [x[i][0], x[i][-1]]
                zyrest = [-20, -20]
                wyrest = [0, 0]
                plt.plot(zxrest, zyrest, 'k--', label='resting potential')
                plt.plot(zxrest, wyrest, 'm--', label='value of decay')
                plt.plot(x_rest[i], y_rest[i], 'ks', label='spike_rest')
            plt.plot(x_spike[i], y_spike[i], 'rs', label='spike_threshold')

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][0]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'g-',
                                 label="static inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'bs',
                                 label="init value before static inhib")
                        plt.plot(x_temp[0], y_temp_after, 'gs',
                                 label="value after static inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'g-')
                        plt.plot(x_temp[0], y_temp_before, 'bs')
                        plt.plot(x_temp[0], y_temp_after, 'gs')
            except ValueError:
                pass

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][1]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'c-',
                                 label="lateral inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'c*',
                                 label="init value before lateral inhib")
                        plt.plot(x_temp[0], y_temp_after, 'y*',
                                 label="value after lateral inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'c-')
                        plt.plot(x_temp[0], y_temp_before, 'c*')
                        plt.plot(x_temp[0], y_temp_after, 'y*')
            except ValueError:
                pass

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][2]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'r-',
                                 label="topdown inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'r*',
                                 label="init value before topdown inhib")
                        plt.plot(x_temp[0], y_temp_after, 'k*',
                                 label="value after topdown inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'r-')
                        plt.plot(x_temp[0], y_temp_before, 'r*')
                        plt.plot(x_temp[0], y_temp_after, 'k*')
            except ValueError:
                pass

            plt.xlabel('Time (µs) ')
            plt.ylabel('Membrane potential (mV)')
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            # plt.legend()

        if(i != number_of_displays-1):
            x.append([])
            y.append([])
            x_rest.append([])
            y_rest.append([])
            x_spike.append([])
            y_spike.append([])
            excit_x.append([])
            excit_y.append([])
            thresh.append([])

    # plt.tight_layout()
    # plt.show()
    return x, y  


def amount_of_excitation_inhibition(spinet: SpikingNetwork, layer_id, neuron_z, visualize=True, average_neurons=False):
    number_of_displays = len(spinet.stats)
    amounts = 4
    arge = np.arange(0, len(np.array(range(0, number_of_displays))+1)*5, 5)
    arge[0] = 1
    x = [[]]
    y = [[]]
    plt.figure(1)
    if(not average_neurons):
        for i in range(amounts):
            for j in range(number_of_displays):
                x[i].append(j+1)
                y[i].append(spinet.stats[j][str(j)][layer_id][str(
                    layer_id)][neuron_z][0]["amount_of_events"][i])
            if(visualize):
                if(i == 0 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'r-', label="excitation")
                if(i == 1 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'y-', label="static inhibition")
                if(i == 2 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'k-', label="lateral inhibition")
                if(i == 3 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'b-', label="topdown inhibition")
                plt.xlabel('length in pixels ')
                plt.ylabel('amount of events')
                plt.legend(bbox_to_anchor=(1.1, 1.05))
            x.append([])
            y.append([])

    else:
        deviation = [[]]
        for i in range(amounts):
            for j in range(number_of_displays):
                x[i].append(j+1)
                avg = 0
                std_dev = 0
                max_num = 0
                for value in range(len(spinet.stats[j][str(j)][layer_id][str(layer_id)])):
                    if (len(spinet.stats[j][str(j)][layer_id][str(layer_id)][value][0]["amount_of_events"]) != 0):
                        max_num += 1
                        avg += spinet.stats[j][str(j)][layer_id][str(
                            layer_id)][value][0]["amount_of_events"][i]
                avg /= max_num
                for value in range(len(spinet.stats[j][str(j)][layer_id][str(layer_id)])):
                    if (len(spinet.stats[j][str(j)][layer_id][str(layer_id)][value][0]["amount_of_events"]) != 0):
                        std_dev += (spinet.stats[j][str(j)][layer_id][str(
                            layer_id)][value][0]["amount_of_events"][i]-avg)**2
                    std_dev /= max_num
                    std_dev = np.sqrt(std_dev)
                y[i].append(avg)
                deviation[i].append(std_dev)
            if(visualize):
                if(i == 0 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'r-', label="excitation") #/max(y[i])
                    print(max(y[i]))
                if(i == 1 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'y-', label="static inhibition")
                if(i == 2 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'k-', label="lateral inhibition")
                    for cter, value in enumerate(x[i]):
                        plt.plot([arge[cter], arge[cter]], [
                                 (y[i][cter]+deviation[i][cter])/max(y[i]), (y[i][cter]-deviation[i][cter])/max(y[i])], 'g-')
                if(i == 3 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'b-', label="topdown inhibition")
                plt.xlabel('length in pixels ')
                plt.ylabel('amount of events')
                plt.legend(bbox_to_anchor=(1.1, 1.05))
            x.append([])
            y.append([])
            deviation.append([])
    plt.xticks(arge)
    plt.title("Amount of excitation and inhibition")
    return x, y

def visualize_inhibition_weights(spinet: SpikingNetwork, layer_id, neuron_id):
    lateral_weights = []
    true_z = []
    avg = []
    for z in range(spinet.l_shape[layer_id][2]):
        lateral_weights.append(spinet.neurons[layer_id][neuron_id].weights_li)
        true_z.append(z)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        counter = 0
        wi = 0
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for value_x in range(x_neur-range_x, x_neur+range_x+1):
            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                    x.append(value_x)
                    y.append(value_y)
        initct = 0
        for value in lateral_weights:
            for value_ in value:
                wi += value_
                counter += 1
                if(counter > spinet.l_shape[layer_id][2]):
                    avg.append(wi/spinet.l_shape[layer_id][2])
                    wi = 0
                    counter = 0
            initct+=1
        fig = plt.figure(figsize=(20, 20), dpi=80)
        max_ = max(avg)
        avg = np.array(avg)/max_
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Blues')
        rect = []
        for var in range(len(x)):
            rect.append(matplotlib.patches.Rectangle(
                (x[var], y[var]), space, space, color=Blues(avg[var])))
            ax.add_patch(rect[var])
        rect.append(matplotlib.patches.Rectangle(
            (x_neur, y_neur), space, space, color='k'))
        ax.add_patch(rect[-1])
        if(x_neur == 0 and y_neur == 0):
            plt.xlim([x_neur, x[-1]+space])
            plt.ylim([y_neur, y[-1]+space])
        elif(x_neur+space >= x[-1]+space and y_neur+space >= y[-1]+space):
            plt.xlim([x[0], x_neur+space])
            plt.ylim([y[0], y_neur+space])
        else:
            plt.xlim([x[0], x[-1]+space])
            plt.ylim([y[0], y[-1]+space])
        cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
        cmapp.set_clim(0, max_)
        plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
        plt.gca().set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.axis('off')
        plt.show()
    return avg

#NO LONGER USED
def visualize_td_inhibition(spinet: SpikingNetwork, layer_id, neuron_id):
    weights_tdi = spinet.neurons[layer_id][neuron_id].weights_tdi
    depth = spinet.l_shape[layer_id+1][2]
    max_ = max(weights_tdi)
    weights_tdi = weights_tdi / max_
    space = 1
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    ordinate = np.linspace(0,len(weights_tdi)/depth -1,int(len(weights_tdi)/depth))
    y_i=0
    x_i=0
    for i, value in enumerate(weights_tdi):
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        rect.append(matplotlib.patches.Rectangle(
            (x_i, ordinate[y_i]), space, space, color=Blues(value)))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    cmapp.set_clim(0, max_)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.xlim([0, depth+space])
    plt.ylim([ordinate[0], ordinate[-1]+space])
    plt.axis('off')
    plt.show()

#OLD_VER    
def visualize_td_sum_inhibition(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, sequence):
    number_of_displays=len(spinet.stats)
    max_tdi = 0
    sum_weights_tdi = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
    for seq in range(number_of_displays):
        sum_seq = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            try:
                sum_seq += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                if(seq==sequence):
                    sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
            except:
                ValueError
        if(max_tdi < np.max(sum_seq)):
            max_tdi= np.max(sum_seq)
    sum_weights_tdi /=len(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)])
    space = 1
    #print(sum_weights_tdi.shape)
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    max_tdi /= len(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)])
    depth = spinet.l_shape[layer_id+1][2]
    ordinate = np.linspace(0,len(spinet.neurons[layer_id][neuron_id].out_connections)/depth -1,int(len(spinet.neurons[layer_id][neuron_id].out_connections)/depth))
    y_i=0
    x_i=0
    for i in range(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1]):
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        rect.append(matplotlib.patches.Rectangle(
            (x_i, ordinate[y_i]), space, space, color=Blues(sum_weights_tdi[y_i][x_i])))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    cmapp.set_clim(0, max_tdi)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_tdi/4, max_tdi/2, 3*max_tdi/4, max_tdi))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.xlim([0, depth+space])
    plt.ylim([ordinate[0], ordinate[-1]+space])
    plt.axis('off')
    plt.show()
    
#OLD_VER    
def visualize_sum_inhibition_weights(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, sequence):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    sum_weights_lat = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
    #sum_weights_lat = spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][neuron_z][2]["sum_inhib_weights"][1]
    max_lat = 0
    number_of_displays = len(spinet.stats)
    for seq in range(number_of_displays):
        if(seq<5):
            continue
        sum_seq = np.zeros((np.array(spinet.stats[seq][str(seq)][layer_id][str(
            layer_id)][0][2]["sum_inhib_weights"][1])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_seq += np.array(spinet.stats[seq][str(seq)][layer_id]
                                [str(layer_id)][r][2]["sum_inhib_weights"][1])
            if(seq == sequence):
                sum_weights_lat += np.array(spinet.stats[seq][str(
                    seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
        if(max_lat < max(sum_seq)):
            max_lat = max(sum_seq)
            #print(seq)
    sum_weights_lat /= len(spinet.stats[sequence]
                           [str(sequence)][layer_id][str(layer_id)])
    max_lat /= len(spinet.stats[sequence]
                   [str(sequence)][layer_id][str(layer_id)])

    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        it = 0
        for value_x in range(x_neur-range_x, x_neur+range_x+1):
            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                if(it < len(sum_weights_lat) and (value_x != x_neur or value_y != y_neur)):
                    value = sum_weights_lat[it]
                    it += 1
                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                    x.append(value_x)
                    y.append(value_y)
                    wi = value
                    avg.append(wi)
        fig = plt.figure(figsize=(20, 20), dpi=80)
        #max_=max(avg)
        max_ = max_lat
        avg = np.array(avg)/max_
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Greys')
        rect = []
        print(len(x))
        for var in range(len(x)):
            rect.append(matplotlib.patches.Rectangle(
                (x[var], y[var]), space, space, color=Blues(avg[var])))
            ax.add_patch(rect[var])
        rect.append(matplotlib.patches.Rectangle(
            (x_neur, y_neur), space, space, color='k'))
        ax.add_patch(rect[-1])
        if(x_neur == 0 and y_neur == 0):
            ax.set_xlim([x_neur, x[-1]+space])
            ax.set_ylim([y_neur, y[-1]+space])
        elif(x_neur+space >= x[-1]+space and y_neur+space >= y[-1]+space):
            ax.set_xlim([x[0], x_neur+space])
            ax.set_ylim([y[0], y_neur+space])
        else:
            ax.set_xlim([x[0], x[-1]+space])
            ax.set_ylim([y[0], y[-1]+space])
        cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
        cmapp.set_clim(0, max_)
        plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
        plt.gca().set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.axis('off')
        plt.show()
    return avg


def visualize_histogram_of_potentials(spinet: SpikingNetwork, layer_id, neuron_z):
    number_of_displays = len(spinet.stats)
    x, y, _, _ = visualize_potentials(spinet, layer_id, neuron_z, False)
    for i in range(number_of_displays):
        plt.figure(i+1)
        fig = plt.figure(i+1)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.hist(y[i], bins=100)
        plt.xlim(-80, 31)
        plt.xlabel('potentials')
        plt.ylabel('histogram distribution')

#OLD_VER
def visualize_evolution_of_inhibition(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, x_neur_to_look, y_neur_to_look, norm_factor=1):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    number_of_displays = len(spinet.stats)
    sum_weights_lat_total = []
    for seq in range(number_of_displays):
        sum_weights_lat = np.zeros((np.array(spinet.stats[seq][str(
            seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_weights_lat += np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
        sum_weights_lat /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
        sum_weights_lat_total.append(sum_weights_lat)

    # for sequence in range(number_of_displays):
      #  sum_weights_lat = spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][neuron_z][2]["sum_inhib_weights"][1]
       # sum_weights_lat_total.append(sum_weights_lat)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for sequence in range(number_of_displays):
            it = 0
            for value_x in range(x_neur-range_x, x_neur+range_x+1):
                for value_y in range(y_neur-range_y, y_neur+range_y+1):
                    if((it < len(sum_weights_lat_total[sequence])) and (value_x != x_neur or value_y != y_neur)):
                        value = sum_weights_lat_total[sequence][it]
                        it += 1
                    if(value_x == x_neur and value_y == y_neur):
                        value = 0
                    if((value_x == x_neur_to_look and value_y == y_neur_to_look)):
                        wi = value
                        avg.append(wi)
        if(norm_factor == 1):
            arge = np.arange(
                0, len(np.array(range(0, number_of_displays))+1)*5, 5)
            #arge[0] = 1
            ax = plt.plot(
                arge, avg, 'k-', label="evolution of the amount of inhibition sent by the neuron")
        else:
            avg = (np.array(avg)/max(avg)) * norm_factor
            ax = plt.plot(range(1, number_of_displays+1), avg, 'k-',
                          label="evolution of the amount of inhibition sent by the neuron")
    return avg

#NO LONGER USED
def visualize_total_inhibition_evolution(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, norm_factor=1):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    number_of_displays = len(spinet.stats)
    sum_weights_lat_total = []
    for seq in range(number_of_displays):
        try:
            sum_weights_lat = np.zeros((np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
            for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
                """if(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1]).shape[0] == 80):
                    print(r)
                    print(r)"""
                sum_weights_lat += np.array(spinet.stats[seq][str(
                    seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
            sum_weights_lat /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
            sum_weights_lat_total.append(sum_weights_lat)
        except: 
            sum_weights_lat_total.append(np.zeros((np.array(spinet.stats[number_of_displays-1][str(
                number_of_displays-1)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape))
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for sequence in range(number_of_displays):
            it = 0
            wi = 0
            for value_x in range(x_neur-range_x, x_neur+range_x+1):
                for value_y in range(y_neur-range_y, y_neur+range_y+1):
                    if((it < len(sum_weights_lat_total[sequence])) and (value_x != x_neur or value_y != y_neur)):
                        value = sum_weights_lat_total[sequence][it]
                        it += 1
                    if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                        wi += value
            avg.append(wi)
        arge = np.arange(1, len(np.array(range(0, number_of_displays)))+1)
        #arge[0] = 1
        if(norm_factor == 1):
            plt.plot(arge, avg, 'r-',
                          label="total amount of lateral inhibition")
        else:
            avg = (np.array(avg)/max(avg)) * norm_factor
            plt.plot(
                arge, avg, 'r-', label="evolution of the amount of inhibition sent by the neuron")
    return avg

#OLD_VER
def visualize_total_tdinhibition_evolution(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z):
    topdown_weights = spinet.neurons[layer_id][neuron_id].weights_tdi
    number_of_displays = len(spinet.stats)
    sum_weights_td_total = []
    for seq in range(number_of_displays):
        sum_weights_td = np.zeros((np.array(spinet.stats[seq][str(
            seq)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_weights_td += np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
        sum_weights_td /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
        sum_weights_td_total.append(sum_weights_td)
    if(len(topdown_weights) != 0):
        avg = []
        wi = 0
        for sequence in range(number_of_displays):
            wi = 0
            for cplex_cell in range(len(sum_weights_td_total[sequence])):
                    for depth in range(spinet.l_shape[layer_id+1][2]):
                        #print(np.array(sum_weights_td_total).shape)
                        wi += sum_weights_td_total[sequence][cplex_cell][depth]
            avg.append(wi)
        arge = np.arange(0, len(np.array(range(0, number_of_displays))))
        #arge[0] = 1
        ax = plt.plot(arge, avg, 'g-',label="total amount of topdown inhibition")
    return avg
  
def suppression_metric(spikes, start_space = 45):
    #give as input spikes with length increasing by 1 pixel each time.
    spikes = np.array(spikes)
    max_spikes = np.max(spikes)
    metric = calculate_metric(max_spikes, spikes[-1])
    return metric

def calculate_metric(max_spike, spike_val):
    v = 100 - (spike_val*100/max_spike) 
    return v

def average_over_orientations(spinet: SpikingNetwork, case, save_folder, n_thickness, angles, n_direction, n_simulation, neuron_id, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, allthick = False, other_params=False, n_speed  =3):
    depth_complex = 4
    thresh_cells = n_simulation/2 + 1
    spinet.load_orientations()
    if case==0:
        for thickness in range(1,n_thickness+1):
            print("Thickness {}/{}".format(thickness,n_thickness))
            sim_mean_spikes = []
            sim_mean_lat = []
            sim_mean_td = []
            sim_mean_stat = []
            sim_mean_lat_tuned = []
            sim_mean_td_tuned = []
            sim_mean_stat_tuned = []
            for sim in range(n_simulation):
                print("Simulation {}/{}".format(sim+1,n_simulation))
                angle_mean_spikes = []
                angle_mean_lat = []
                angle_mean_td = []
                angle_mean_stat = []
                angle_mean_stat_tuned = []
                angle_mean_lat_tuned = []
                angle_mean_td_tuned = []
                for angle in angles:
                    for direction in range(n_direction):
                        print("Angle {}° in direction {}".format(angle, direction))
                        spinet.load_statistics_2(thickness, angle, direction, layer_id, sim)
                        y_tot = []
                        cells_number = []
                        for i in range(max_depth):
                            number_of_displays = len(spinet.stats)
                            if(not tuned_ori):
                                for seq in range(number_of_displays):
                                    temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                    try:
                                        if( (temp[0]>thresh).any() ):
                                            ok = True
                                            break
                                        else:
                                            ok = False
                                    except: 
                                        ok = False
                                condition = ok
                            else:
                                theta = np.array(spinet.neurons[layer_id][neuron_id+neuron_z].theta)
                                if(allthick):
                                    theta_3 = theta[2]
                                    theta = np.append(np.array(theta[0]),np.array(theta[1]))
                                    theta = np.append(theta, np.array(theta_3))
                                if(angle!=0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                                elif(angle==0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                            if(condition):
                                cells_number.append(neuron_z)
                                _, y = visualize_potentials(spinet, layer_id, neuron_z, False)
                                spikes_number = []
                                for value in range(number_of_displays):
                                    spikes_number.append(len(np.where(np.array(y[value])>=thresh)[0]))
                                y_tot.append(spikes_number)
                                print("Depth {0} approved (spiked)".format(neuron_z))
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                        mean_val = []
                        for i in range(number_of_displays):
                            mean = 0
                            for j in range(len(y_tot)):
                                mean += y_tot[j][i]
                            if(len(y_tot)!=0):
                                mean/=len(y_tot)
                            else:
                                mean = 0
                            mean_val.append(mean)
                                
                        #if(not other_params):
                        try:
                            avg0_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, untuned_simple = True)     #visualize_total_inhibition_evolution(spinet, layer_id, neuron_id, neuron_z,1)
                            avg1_ = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = max_depth, depth_complex = depth_complex, tuned_simple = True, allthick = True, untuned_complex=True)  #visualize_total_tdinhibition_evolution(spinet, layer_id, neuron_id, neuron_z)
                            avg2_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, untuned_simple = True, inhibition_type=0)
                        
                            avg0bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False)  
                            avg1bis = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = max_depth, depth_complex = depth_complex, tuned_simple = True, allthick = True, untuned_complex=False)
                            avg2bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, inhibition_type=0)  
                        #else:
                            #to complete everywhereusing data_analysis_inhibition()
                            
                        except:
                            avg0_ = np.zeros((number_of_displays))
                            avg1_ = np.zeros((number_of_displays))
                            avg2_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, untuned_simple = True, inhibition_type=0)
                            
                            avg0bis = np.zeros((number_of_displays))
                            avg1bis = np.zeros((number_of_displays))
                            avg2bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, inhibition_type=0)
                            
                        angle_mean_spikes.append(mean_val)
                        angle_mean_lat.append(avg0_)
                        angle_mean_td.append(avg1_)
                        angle_mean_stat.append(avg2_)
                        
                        angle_mean_stat_tuned.append(avg2bis)
                        angle_mean_lat_tuned.append(avg0bis)
                        angle_mean_td_tuned.append(avg1bis)
                
                sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                sim_mean_lat.append(np.mean(angle_mean_lat, axis=0))
                sim_mean_td.append(np.mean(angle_mean_td, axis=0))
                sim_mean_stat.append(np.mean(angle_mean_stat, axis=0))
                sim_mean_lat_tuned.append(np.mean(angle_mean_lat_tuned, axis=0))
                sim_mean_td_tuned.append(np.mean(angle_mean_td_tuned, axis=0))
                sim_mean_stat_tuned.append(np.mean(angle_mean_stat_tuned, axis = 0))
                
            avg_spikes = np.mean(sim_mean_spikes, axis=0)
            avg_lat = np.mean(sim_mean_lat, axis=0)
            avg_td = np.mean(sim_mean_td, axis=0)
            avg_stat = np.mean(sim_mean_stat, axis=0)
            avg_lat_tuned = np.mean(sim_mean_lat_tuned, axis=0)
            avg_td_tuned = np.mean(sim_mean_td_tuned, axis=0)
            avg_stat_tuned = np.mean(sim_mean_stat_tuned, axis=0)
            
            std_spikes_err = []
            std_lat_err = []
            std_td_err = []
            std_stat_err = []
            std_lat_tuned_err = []
            std_td_tuned_err = []
            std_stat_tuned_err = []
            
            for i in range(number_of_displays):
                std_spikes = 0
                std_lat = 0
                std_td = 0
                std_stat = 0
                std_lat_tuned = 0
                std_td_tuned = 0
                std_stat_tuned = 0
                for j in range(n_simulation):
                    std_spikes+=(sim_mean_spikes[j][i]-avg_spikes[i])**2
                    std_lat+=(sim_mean_lat[j][i]-avg_lat[i])**2
                    std_td+=(sim_mean_td[j][i]-avg_td[i])**2
                    std_stat+=(sim_mean_stat_tuned[j][i]-avg_stat[i])**2
                    std_lat_tuned+=(sim_mean_lat_tuned[j][i]-avg_lat_tuned[i])**2
                    std_td_tuned+=(sim_mean_td_tuned[j][i]-avg_td_tuned[i])**2
                    std_stat_tuned+=(sim_mean_stat_tuned[j][i]-avg_stat_tuned[i])**2
                    
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
                std_lat_err.append(np.sqrt(std_lat/(n_simulation-1))/n_simulation)
                std_td_err.append(np.sqrt(std_td/(n_simulation-1))/n_simulation)
                std_stat_err.append(np.sqrt(std_stat/(n_simulation-1))/n_simulation)
                std_lat_tuned_err.append(np.sqrt(std_lat_tuned/(n_simulation-1))/n_simulation)
                std_td_tuned_err.append(np.sqrt(std_td_tuned/(n_simulation-1))/n_simulation)
                std_stat_tuned_err.append(np.sqrt(std_stat_tuned/(n_simulation-1))/n_simulation)
                
                
            np.save(save_folder+str(thickness)+"/orientations_average/" + "spikes_avg.npy", np.array(avg_spikes))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lat_avg.npy", np.array(avg_lat))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "td_avg.npy", np.array(avg_td))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stat_avg.npy", np.array(avg_stat))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lattuned_avg.npy", np.array(avg_lat_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "tdtuned_avg.npy", np.array(avg_td_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stattuned_avg.npy", np.array(avg_stat_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "spikes_stderr.npy", np.array(std_spikes_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lat_stderr.npy", np.array(std_lat_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "td_stderr.npy", np.array(std_td_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lattuned_stderr.npy", np.array(std_lat_tuned_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "tdtuned_stderr.npy", np.array(std_td_tuned_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stattuned_stderr.npy", np.array(std_stat_tuned_err))
            
            
            
    elif case==1:
        for thickness in range(1,n_thickness+1):
            for angle in angles:
                for direction in range(n_direction):
                    sim_mean_spikes = []
                    sim_mean_lat = []
                    sim_mean_td = []
                    sim_cells_number = []
                    for sim in range(n_simulation):
                        spinet.load_statistics_2(thickness, angle, direction, layer_id, sim)
                        y_tot = []
                        cells_number = []
                        for i in range(max_depth):
                            number_of_displays = len(spinet.stats)
                            if(not tuned_ori):
                                for seq in range(number_of_displays):
                                    temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                    try:
                                        if( (temp[0]>thresh).any() ):
                                            ok = True
                                            break
                                        else:
                                            ok = False
                                    except: 
                                        ok = False
                                condition = ok
                            else:
                                theta = np.array(spinet.neurons[layer_id][neuron_id+neuron_z].theta)
                                if(allthick):
                                    theta_3 = theta[2]
                                    theta = np.append(np.array(theta[0]),np.array(theta[1]))
                                    theta = np.append(theta, np.array(theta_3))
                                if(angle!=0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                                        if(condition):
                                            for seq in range(number_of_displays):
                                                temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                                try:
                                                    if( (temp[0]>=thresh).any() ):
                                                        ok = True
                                                        break
                                                    else:
                                                        ok = False
                                                except: 
                                                    ok = False
                                            condition = ok
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                                elif(angle==0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                                        if(condition):
                                            for seq in range(number_of_displays):
                                                temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                                try:
                                                    if( (temp[0]>=thresh).any() ):
                                                        ok = True
                                                        break
                                                    else:
                                                        ok = False
                                                except: 
                                                    ok = False
                                            condition = ok
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                            if(condition):
                                cells_number.append(neuron_z)
                                _, y = visualize_potentials(spinet, layer_id, neuron_z, False)
                                spikes_number = []
                                for value in range(number_of_displays):
                                    spikes_number.append(len(np.where(np.array(y[value])>=thresh)[0]))
                                y_tot.append(spikes_number)
                                print("Depth {0} approved (spiked)".format(neuron_z))
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                        mean_val = []
                        for i in range(number_of_displays):
                            mean = 0
                            for j in range(len(y_tot)):
                                mean += y_tot[j][i]
                            mean/=len(y_tot)
                            mean_val.append(mean)
                                
                        avg0_ = visualize_total_inhibition_evolution(spinet, layer_id, neuron_id, neuron_z,1)
                        avg1_ = visualize_total_tdinhibition_evolution(spinet, layer_id, neuron_id, neuron_z)
                    
                        sim_mean_spikes.append(mean_val)
                        sim_mean_lat.append(avg0_)
                        sim_mean_td.append(avg1_)
                        sim_cells_number.append(cells_number)
                
                    avg_spikes = np.mean(sim_mean_spikes, axis=0)
                    avg_lat = np.mean(sim_mean_lat, axis=0)
                    avg_td = np.mean(sim_mean_td, axis=0)
                    std_spikes_err = []
                    std_lat_err = []
                    std_td_err = []
                    for i in range(number_of_displays):
                        std_spikes = 0
                        std_lat = 0
                        std_td = 0
                        for j in range(n_simulation):
                            std_spikes+=(sim_mean_spikes[j][i]-avg_spikes[i])**2
                            std_lat+=(sim_mean_lat[j][i]-avg_lat[i])**2
                            std_td+=(sim_mean_td[j][i]-avg_td[i])**2
                        std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
                        std_lat_err.append(np.sqrt(std_lat/(n_simulation-1))/n_simulation)
                        std_td_err.append(np.sqrt(std_td/(n_simulation-1))/n_simulation)
                
                    avg_cells = []
                    diff_values = np.unique(sim_cells_number)
                    diff_values = np.sort(diff_values)
                    cter_values = np.zeros((len(diff_values)))
                    flattened = np.array(sim_cells_number).flatten()
                    for i, value in enumerate(diff_values):
                        for val in flattened:
                            if(val==value):
                                cter_values[i]+=1
                    index = np.where(cter_values>thresh_cells)
                    for i in index[0]:
                        avg_cells.append(diff_values[i])
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "spikes_avg.npy", np.array(avg_spikes))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "lat_avg.npy", np.array(avg_lat))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "td_avg.npy", np.array(avg_td))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "lat_stderr.npy", np.array(std_lat_err))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "td_stderr.npy", np.array(std_td_err))
                    np.save(save_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "responsivecells_avg.npy", np.array(avg_cells))
            
    elif case==2:
        for speed in range(0,n_speed):
            print("Speed {}/{}".format(speed+1,n_speed))
            for thickness in range(1,n_thickness+1):
                print("Thickness {}/{}".format(thickness,n_thickness))
                sim_mean_spikes = []
                sim_mean_lat = []
                sim_mean_td = []
                sim_mean_stat = []
                sim_mean_lat_tuned = []
                sim_mean_td_tuned = []
                sim_mean_stat_tuned = []
                sim_cells_number = []
                for sim in range(n_simulation):
                    print("Simulation {}/{}".format(sim+1,n_simulation))
                    angle_mean_spikes = []
                    angle_mean_lat = []
                    angle_mean_td = []
                    angle_mean_stat = []
                    angle_mean_stat_tuned = []
                    angle_mean_lat_tuned = []
                    angle_mean_td_tuned = []
                    for angle in angles:
                        for direction in range(n_direction):
                            print("Angle {}° in direction {}".format(angle, direction))
                            spinet.load_statistics_2(thickness, angle, direction, layer_id, sim, separate_speed=True, speed = speed)
                            y_tot = []
                            cells_number = []
                            for i in range(max_depth):
                                number_of_displays = len(spinet.stats)
                                if(not tuned_ori):
                                    for seq in range(number_of_displays):
                                        temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                        try:
                                            if( (temp[0]>thresh).any() ):
                                                ok = True
                                                break
                                            else:
                                                ok = False
                                        except: 
                                            ok = False
                                    condition = ok
                                else:
                                    theta = np.array(spinet.neurons[layer_id][neuron_id+neuron_z].theta)
                                    if(allthick):
                                        theta_3 = theta[2]
                                        theta = np.append(np.array(theta[0]),np.array(theta[1]))
                                        theta = np.append(theta, np.array(theta_3))
                                    if(angle!=0):
                                        if(allthick):
                                            condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                                            """if(condition):
                                                for seq in range(number_of_displays):
                                                    temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                                    try:
                                                        if( (temp[0]>=thresh).any() ):
                                                            ok = True
                                                            break
                                                        else:
                                                            ok = False
                                                    except: 
                                                        ok = False
                                                condition = ok"""
                                        else:
                                            condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                                    elif(angle==0):
                                        if(allthick):
                                            condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                                            """if(condition):
                                                for seq in range(number_of_displays):
                                                    temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                                    try:
                                                        if( (temp[0]>=thresh).any() ):
                                                            ok = True
                                                            break
                                                        else:
                                                            ok = False
                                                    except: 
                                                        ok = False
                                                condition = ok"""
                                        else:
                                            condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                                if(condition):
                                    cells_number.append(neuron_z)
                                    _, y = visualize_potentials(spinet, layer_id, neuron_z, False)
                                    spikes_number = []
                                    for value in range(number_of_displays):
                                        spikes_number.append(len(np.where(np.array(y[value])>=thresh)[0]))
                                    y_tot.append(spikes_number)
                                    print("Depth {0} approved (spiked)".format(neuron_z))
                                    neuron_z+=1
                                    if(neuron_z>=max_depth):
                                        break
                                else:
                                    neuron_z+=1
                                    if(neuron_z>=max_depth):
                                        break
                            neuron_z = 0
                            mean_val = []
                            for i in range(number_of_displays):
                                mean = 0
                                for j in range(len(y_tot)):
                                    mean += y_tot[j][i]
                                if(len(y_tot)!=0):
                                    mean/=len(y_tot)
                                else:
                                    mean = 0
                                mean_val.append(mean)
                                    
                            #if(not other_params):
                            try:
                                avg0_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False, untuned_simple = True)     #visualize_total_inhibition_evolution(spinet, layer_id, neuron_id, neuron_z,1)
                                avg1_ = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = 64, depth_complex = 16, tuned_simple = True, allthick = True, untuned_complex=True)  #visualize_total_tdinhibition_evolution(spinet, layer_id, neuron_id, neuron_z)
                                avg2_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False, untuned_simple = True, inhibition_type=0)
                            
                                avg0bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False)  
                                avg1bis = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = 64, depth_complex = 16, tuned_simple = True, allthick = True, untuned_complex=False)
                                avg2bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False, inhibition_type=0)  
                            #else:
                                #to complete everywhereusing data_analysis_inhibition()
                                
                            except:
                                avg0_ = np.zeros((number_of_displays))
                                avg1_ = np.zeros((number_of_displays))
                                try:
                                    avg2_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False, untuned_simple = True, inhibition_type=0)
                                except:
                                    avg2_ = np.zeros((number_of_displays))
                                avg0bis = np.zeros((number_of_displays))
                                avg1bis = np.zeros((number_of_displays))
                                try:
                                    avg2bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = 64, tuned_simple = True, allthick = True, without = False, inhibition_type=0)
                                except:
                                    avg2bis = np.zeros((number_of_displays))

                            angle_mean_spikes.append(mean_val)
                            angle_mean_lat.append(avg0_)
                            angle_mean_td.append(avg1_)
                            angle_mean_stat.append(avg2_)
                            
                            angle_mean_stat_tuned.append(avg2bis)
                            angle_mean_lat_tuned.append(avg0bis)
                            angle_mean_td_tuned.append(avg1bis)
                    
                    sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                    sim_mean_lat.append(np.mean(angle_mean_lat, axis=0))
                    sim_mean_td.append(np.mean(angle_mean_td, axis=0))
                    sim_mean_stat.append(np.mean(angle_mean_stat, axis=0))
                    sim_mean_lat_tuned.append(np.mean(angle_mean_lat_tuned, axis=0))
                    sim_mean_td_tuned.append(np.mean(angle_mean_td_tuned, axis=0))
                    sim_mean_stat_tuned.append(np.mean(angle_mean_stat_tuned, axis = 0))
                    
                avg_spikes = np.mean(sim_mean_spikes, axis=0)
                avg_lat = np.mean(sim_mean_lat, axis=0)
                avg_td = np.mean(sim_mean_td, axis=0)
                avg_stat = np.mean(sim_mean_stat, axis=0)
                avg_lat_tuned = np.mean(sim_mean_lat_tuned, axis=0)
                avg_td_tuned = np.mean(sim_mean_td_tuned, axis=0)
                avg_stat_tuned = np.mean(sim_mean_stat_tuned, axis=0)
                
                std_spikes_err = []
                std_lat_err = []
                std_td_err = []
                std_stat_err = []
                std_lat_tuned_err = []
                std_td_tuned_err = []
                std_stat_tuned_err = []
                
                for i in range(number_of_displays):
                    std_spikes = 0
                    std_lat = 0
                    std_td = 0
                    std_stat = 0
                    std_lat_tuned = 0
                    std_td_tuned = 0
                    std_stat_tuned = 0
                    for j in range(n_simulation):
                        std_spikes+=(sim_mean_spikes[j][i]-avg_spikes[i])**2
                        std_lat+=(sim_mean_lat[j][i]-avg_lat[i])**2
                        std_td+=(sim_mean_td[j][i]-avg_td[i])**2
                        std_stat+=(sim_mean_stat_tuned[j][i]-avg_stat[i])**2
                        std_lat_tuned+=(sim_mean_lat_tuned[j][i]-avg_lat_tuned[i])**2
                        std_td_tuned+=(sim_mean_td_tuned[j][i]-avg_td_tuned[i])**2
                        std_stat_tuned+=(sim_mean_stat_tuned[j][i]-avg_stat_tuned[i])**2
                        
                    std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
                    std_lat_err.append(np.sqrt(std_lat/(n_simulation-1))/n_simulation)
                    std_td_err.append(np.sqrt(std_td/(n_simulation-1))/n_simulation)
                    std_stat_err.append(np.sqrt(std_stat/(n_simulation-1))/n_simulation)
                    std_lat_tuned_err.append(np.sqrt(std_lat_tuned/(n_simulation-1))/n_simulation)
                    std_td_tuned_err.append(np.sqrt(std_td_tuned/(n_simulation-1))/n_simulation)
                    std_stat_tuned_err.append(np.sqrt(std_stat_tuned/(n_simulation-1))/n_simulation)
            
                avg_cells = []
                diff_values = np.unique(sim_cells_number)
                diff_values = np.sort(diff_values)
                cter_values = np.zeros((len(diff_values)))
                flattened = np.array(sim_cells_number).flatten()
                for i, value in enumerate(diff_values):
                    for val in flattened:
                        if(val==value):
                            cter_values[i]+=1
                index = np.where(cter_values>thresh_cells)
                for i in index[0]:
                    avg_cells.append(diff_values[i])
                    
                    
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) + "/orientations_average/" + "spikes_avg.npy", np.array(avg_spikes))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_avg.npy", np.array(avg_lat))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_avg.npy", np.array(avg_td))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "stat_avg.npy", np.array(avg_stat))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_avg.npy", np.array(avg_lat_tuned))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "tdtuned_avg.npy", np.array(avg_td_tuned))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "stattuned_avg.npy", np.array(avg_stat_tuned))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_stderr.npy", np.array(std_spikes_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_stderr.npy", np.array(std_lat_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_stderr.npy", np.array(std_td_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_stderr.npy", np.array(std_lat_tuned_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "tdtuned_stderr.npy", np.array(std_td_tuned_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "stattuned_stderr.npy", np.array(std_stat_tuned_err))
        
                
def get_average_nb_events(spinet: SpikingNetwork, case, save_folder, n_thickness, angles, n_direction, n_simulation, neuron_id, layer_id = 0, neuron_z = 0, max_depth = 64, thresh = 30, tuned_ori=False, allthick = False, other_params=False, n_speed  =3):
    
    if case==0:
        for thickness in range(3,n_thickness+1):
            sim_mean_events = []
            print("Thickness {}/{}".format(thickness,n_thickness))
            for sim in range(n_simulation):
                angle_mean_events = []
                print("Simulation {}/{}".format(sim+1,n_simulation))
                for angle in angles:
                    for direction in range(n_direction):
                        print("Angle {}° in direction {}".format(angle, direction))
                        spinet.load_statistics_2(thickness, angle, direction, layer_id, sim)
                        number_of_displays = len(spinet.stats)
                        cells_nb_events=[[]]
                        neuron_z = 0
                        once = True
                        for i in range(number_of_displays):
                            for j in range(4):
                                try:
                                    cells_nb_events[j].append(spinet.stats[i][str(i)][layer_id][str(
                                        layer_id)][neuron_z][0]["amount_of_events"][j])
                                except:
                                    cells_nb_events[j].append(0)
                                if(once):
                                    cells_nb_events.append([])
                                    if(j==2):
                                        once = False
    
                        angle_mean_events.append(cells_nb_events)
                
                sim_mean_events.append(np.mean(angle_mean_events, axis=0))
                
            avg_events = np.mean(sim_mean_events, axis=0)
            
            std_events_excit_err = []
            std_events_li_err = []
            std_events_tdi_err = []
            std_events_stat_err = []
            for k in range(4):
                for i in range(number_of_displays):
                    std_events = 0
                    for j in range(n_simulation):
                        std_events+=(sim_mean_events[j][k][i]-avg_events[k][i])**2
                if(k==0):    
                    std_events_excit_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                elif(k==1):
                    std_events_li_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                elif(k==2):
                    std_events_tdi_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                elif(k==3):
                    std_events_stat_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                
                
            np.save(save_folder+str(thickness)+ "/orientations_average/" + "events_avg.npy", np.array(avg_events))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "events_excit_err_avg.npy", np.array(std_events_excit_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "events_li_err_avg.npy", np.array(std_events_li_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "events_tdi_err_avg.npy", np.array(std_events_tdi_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "events_stat_err_avg.npy", np.array(std_events_stat_err))
    
    elif case==1:
        for speed in range(0,n_speed):
            avg_events = []
            print("Speed {}/{}".format(speed+1,n_speed))
            for thickness in range(1,n_thickness+1):
                sim_mean_events = []
                print("Thickness {}/{}".format(thickness,n_thickness))
                for sim in range(n_simulation):
                    angle_mean_events = []
                    print("Simulation {}/{}".format(sim+1,n_simulation))
                    for angle in angles:
                        for direction in range(n_direction):
                            print("Angle {}° in direction {}".format(angle, direction))
                            spinet.load_statistics_2(thickness, angle, direction, layer_id, sim, separate_speed=True, speed = speed)
                            number_of_displays = len(spinet.stats)
                            cells_nb_events=[[]]
                            neuron_z = 0
                            once = True
                            for i in range(number_of_displays):
                                for j in range(4):
                                    try:
                                        cells_nb_events[j].append(spinet.stats[i][str(i)][layer_id][str(
                                            layer_id)][neuron_z][0]["amount_of_events"][j])
                                    except:
                                        cells_nb_events[j].append(0)
                                    if(once):
                                        cells_nb_events.append([])
                                        if(j==2):
                                            once = False
        
                            angle_mean_events.append(cells_nb_events)
                    
                    sim_mean_events.append(np.mean(angle_mean_events, axis=0))
                    
                avg_events = np.mean(sim_mean_events, axis=0)
                
                std_events_excit_err = []
                std_events_li_err = []
                std_events_tdi_err = []
                std_events_stat_err = []
                for k in range(4):
                    for i in range(number_of_displays):
                        std_events = 0
                        for j in range(n_simulation):
                            std_events+=(sim_mean_events[j][k][i]-avg_events[k][i])**2
                    if(k==0):    
                        std_events_excit_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                    elif(k==1):
                        std_events_li_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                    elif(k==2):
                        std_events_tdi_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                    elif(k==3):
                        std_events_stat_err.append(np.sqrt(std_events/(n_simulation-1))/n_simulation)
                    
                    
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) + "/orientations_average/" + "events_avg.npy", np.array(avg_events))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "events_excit_err_avg.npy", np.array(std_events_excit_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "events_li_err_avg.npy", np.array(std_events_li_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "events_tdi_err_avg.npy", np.array(std_events_tdi_err))
                np.save(save_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "events_stat_err_avg.npy", np.array(std_events_stat_err))
            

#PLOTTING FUNCTIONS FOR DIVERSE GRAPHS  
def averaged_graph2(load_folders, case, with_speed = False, n_thickness = 3, thickness = 1, speed = 0, n_speeds = 3, n_displays = 55, avg = False, start_space = 45, space = 3, n_simulation = 5):
            
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    if(case==0): #Plot spikes
        if(len(load_folders) > 1 or avg): #plot spikes of different trials (only lat, only td, both, no inhib etc.) ; works also for speeds.
            avg_spikes = []
            avg_std_err = []
            for i in range(len(load_folders)):
                avg_spikes.append([])
                avg_std_err.append([])
                if(with_speed):
                    for j in range(n_speeds):
                        avg_spikes[i].append([])
                        avg_std_err[i].append([])
            for thness in range(1,n_thickness+1):
                for i, folder in enumerate(load_folders):
                    if(not with_speed):
                        avg_spikes[i].append(np.load(folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
                        avg_std_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                    else:
                        for j in range(n_speeds):
                            avg_spikes[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_avg.npy"))
                            avg_std_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
            
            if(with_speed):
                avg_spikes = np.mean(avg_spikes, axis = 2)
                avg_std_err = np.mean(avg_std_err, axis = 2)
            avg_spikes = np.mean(avg_spikes, axis=1)
            avg_std_err = np.mean(avg_std_err, axis=1)
            
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Average number of spikes")
            ax.set_title("Evolution of spikes for different cases")
            for i in range(len(load_folders)):
                ax.plot(arge, avg_spikes[i], colors[i]+"-", label="folder number " + str(i))
                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_spikes[i][val]-avg_std_err[i][val],avg_spikes[i][val]+ avg_std_err[i][val]],colors[i]+"--")
            plt.xticks(arge)
            fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
            plt.show()
        else:
            avg_spikes = []
            avg_std_err = []
            
            arge = np.arange(1,start_space+1)
            diff = n_displays- start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
                
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Average number of spikes")
            ax.set_title("Evolution of spikes for different cases")
            
            if(with_speed):
                for j in range(n_speeds):
                    avg_spikes.append([])
                    avg_std_err.append([])
                    
                    avg_spikes[j].append(np.load(load_folders[0]+str(thickness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_std_err[j].append(np.load(load_folders[0]+str(thickness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                
                    ax.plot(arge,avg_spikes[j], colors[j]+"-", label="speed number " + str(j))
                    for val in range(len(arge)):
                        ax.plot( [arge[val], arge[val]], [avg_spikes[j][val]-avg_std_err[j][val],avg_spikes[j][val]+ avg_std_err[j][val]],colors[j]+"--")
            else:
                avg_spikes.append(np.load(load_folders[0]+str(thickness)+"/orientations_average/" + "spikes_avg.npy"))
                avg_std_err.append(np.load(load_folders[0]+str(thickness)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                ax.plot(arge,avg_spikes, colors[0]+"-", label="thickness = " + str(thickness))
                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_spikes[val]-avg_std_err[val],avg_spikes[val]+ avg_std_err[val]],colors[0]+"--")
            plt.xticks(arge)
            fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
            plt.show()
    elif(case==1): #plot inhibitions
        if(len(load_folders) > 1 or avg): #plot spikes of different trials (only lat, only td, both, no inhib etc.) ; works also for speeds.
            avg_lat = []
            avg_td = []
            avg_stat = []
            
            avg_lat_err = []
            avg_td_err = []
            avg_stat_err = []
            
            for i in range(len(load_folders)):
                avg_lat.append([])
                avg_td.append([])
                avg_stat.append([])
                
                avg_lat_err.append([])
                avg_td_err.append([])
                avg_stat_err.append([])
                
                if(with_speed):
                    for j in range(n_speeds):
                        avg_lat[i].append([])
                        avg_td[i].append([])
                        avg_stat[i].append([])
                        
                        avg_lat_err[i].append([])
                        avg_td_err[i].append([])
                        avg_stat_err[i].append([])
                        
            for thness in range(1,n_thickness+1):
                for i, folder in enumerate(load_folders):
                    if(not with_speed):
                        avg_lat[i].append(np.load(folder+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                        avg_td[i].append(np.load(folder+str(thness)+"/orientations_average/" + "td_avg.npy"))
                        avg_stat[i].append(np.load(folder+str(thness)+"/orientations_average/" + "stat_avg.npy"))
                        
                        avg_lat_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                        avg_td_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                        avg_stat_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                    else:
                        for j in range(n_speeds):
                            
                            avg_lat[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_avg.npy"))
                            avg_td[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_avg.npy"))
                            avg_stat[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_avg.npy"))
                            
                            avg_lat_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                            avg_td_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                            avg_stat_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
            
            if(with_speed):
                avg_lat = np.mean(avg_lat, axis = 2)
                avg_td = np.mean(avg_td, axis = 2)
                avg_stat = np.mean(avg_stat, axis=2)
                
                avg_lat_err = np.mean(avg_lat_err, axis = 2)
                avg_td_err = np.mean(avg_td_err, axis = 2)
                avg_stat_err = np.mean(avg_stat_err, axis=2)
                
            avg_lat = np.mean(avg_lat, axis = 0)
            avg_td = np.mean(avg_td, axis = 0)
            avg_stat = np.mean(avg_stat, axis=0)
            
            avg_lat_err = np.mean(avg_lat_err, axis = 0)
            avg_td_err = np.mean(avg_td_err, axis = 0)
            avg_stat_err = np.mean(avg_stat_err, axis=0)
            
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Amount of inhibition")
            ax.set_title("Evolution of spikes for different cases")
            for i in range(len(load_folders)):
                ax.plot(arge, avg_lat[i], colors[i]+"-", label="folder number " + str(i))
                ax.plot(arge, avg_td[i], colors[i]+"o")
                ax.plot(arge, avg_stat[i], colors[i]+"*")

                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_lat[i][val]-avg_lat_err[i][val],avg_lat[i][val]+ avg_lat_err[i][val]],colors[i]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_td[i][val]-avg_td_err[i][val],avg_td[i][val]+ avg_td_err[i][val]],colors[i]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_stat[i][val]-avg_stat_err[i][val],avg_stat[i][val]+ avg_stat_err[i][val]],colors[i]+"--")
            plt.xticks(arge)
            fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
            plt.show()
            
        else:
            avg_lat = []
            avg_td = []
            avg_stat = []
            
            avg_lat_err = []
            avg_td_err = []
            avg_stat_err = []
            
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Average number of spikes")
            ax.set_title("Evolution of spikes for different cases")
            
            if(with_speed):
                for j in range(n_speeds):
                    avg_lat.append([])
                    avg_td.append([])
                    avg_stat.append([])
                    
                    avg_lat_err.append([])
                    avg_td_err.append([])
                    avg_stat_err.append([])
                    
                    avg_lat[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_avg.npy"))
                    avg_td[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_avg.npy"))
                    avg_stat[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_avg.npy"))
                    
                    avg_lat_err[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                    avg_td_err[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                    avg_stat_err[j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                    
                    ax.plot(arge, avg_lat[j], colors[j]+"-", label="speed number " + str(i))
                    ax.plot(arge, avg_td[j], colors[j]+"o")
                    ax.plot(arge, avg_stat[j], colors[j]+"*")

                    for val in range(len(arge)):
                        ax.plot( [arge[val], arge[val]], [avg_lat[j][val]-avg_lat_err[j][val],avg_lat[j][val]+ avg_lat_err[j][val]],colors[j]+"--")
                        ax.plot( [arge[val], arge[val]], [avg_td[j][val]-avg_td_err[j][val],avg_td[j][val]+ avg_td_err[j][val]],colors[j]+"--")
                        ax.plot( [arge[val], arge[val]], [avg_stat[j][val]-avg_stat_err[j][val],avg_stat[j][val]+ avg_stat_err[j][val]],colors[j]+"--")
                            
            else:
                avg_lat.append(np.load(folder+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                avg_td.append(np.load(folder+str(thness)+"/orientations_average/" + "td_avg.npy"))
                avg_stat.append(np.load(folder+str(thness)+"/orientations_average/" + "stat_avg.npy"))
                
                avg_lat_err.append(np.load(folder+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                avg_td_err.append(np.load(folder+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                avg_stat_err.append(np.load(folder+str(thness)+"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                
                ax.plot(arge, avg_lat, colors[0]+"-", label="thickness = " + str(i))
                ax.plot(arge, avg_td, colors[0]+"o")
                ax.plot(arge, avg_stat, colors[0]+"*")

                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_lat[val]-avg_lat_err[val],avg_lat[val]+ avg_lat_err[val]],colors[0]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_td[val]-avg_td_err[val],avg_td[val]+ avg_td_err[val]],colors[0]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_stat[val]-avg_stat_err[val],avg_stat[val]+ avg_stat_err[val]],colors[0]+"--")
    elif(case==2): #plot all thicknesses in one plot ; not for speeds.
        avg_spikes = []
        avg_std_err = []
        for i in range(n_thickness):
            avg_spikes.append([])
            avg_std_err.append([])
            
            avg_spikes[i].append(np.load(folder+str(i)+"/orientations_average/" + "spikes_avg.npy"))
            avg_std_err[i].append(np.load(folder+str(i)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
        
        arge = np.arange(1,start_space+1)
        diff = n_displays - start_space
        for value in range(1, diff+1):
            arge = np.append(arge, start_space + space * value)
        fig = plt.figure(1)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Length of oriented bars (in pixels)")
        ax.set_ylabel("Average number of spikes")
        ax.set_title("Evolution of spikes for different cases")
        for i in range(n_thickness):
            ax.plot(arge, avg_spikes[i], colors[i]+"-", label="thickness number " + str(i))
            for val in range(len(arge)):
                ax.plot( [arge[val], arge[val]], [avg_spikes[i][val]-avg_std_err[i][val],avg_spikes[i][val]+ avg_std_err[i][val]],colors[i]+"--")
        plt.xticks(arge)
        fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
        plt.show()
    
    
def check_steepness(excit_folder, inhib_folder, no_inhib_folder, thickness, start_space = 45, space = 3, n_simulation = 5):
    avg_spikes_noinhib = []
    avg_spikes_noinhib_err = []
    avg_spikes_inhib = []
    avg_spikes_inhib_err = []
    avg_excit_ev = []
    avg_excit_ev_err = []
    avg_inhib_amount = []
    avg_td_amount = []
    avg_inhib_amount_tuned = []
    avg_td_amount_tuned = []
    avg_inhib_amount_err = []
    
    thickness_max = 3
    for thness in range(1,thickness_max+1):
        
        avg_spikes_noinhib.append(np.load(no_inhib_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_inhib.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_noinhib_err.append(np.load(no_inhib_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
        avg_spikes_inhib_err.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
        avg_excit_ev.append(np.load(excit_folder+str(thness)+"/orientations_average/" + "events_avg.npy"))
        avg_excit_ev_err.append(np.load(excit_folder+str(thness)+"/orientations_average/" + "events_excit_err_avg.npy"))
        avg_inhib_amount.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "lat_avg.npy"))
        avg_td_amount.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "td_avg.npy"))
        avg_inhib_amount_tuned.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "lattuned_avg.npy"))
        avg_td_amount_tuned.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "tdtuned_avg.npy"))
        
        avg_inhib_amount_err.append(np.load(inhib_folder+str(thness)+"/orientations_average/" + "lat_stderr.npy"))

    avg_spikes_noinhib = np.mean(avg_spikes_noinhib, axis = 0)
    avg_spikes_inhib = np.mean(avg_spikes_inhib, axis = 0)
    avg_td_amount = np.mean(avg_td_amount, axis = 0)
    avg_spikes_noinhib_err = np.mean(avg_spikes_noinhib_err, axis = 0)
    avg_spikes_inhib_err = np.mean(avg_spikes_inhib_err, axis = 0)
    avg_excit_ev = np.mean(avg_excit_ev, axis = 0)
    avg_excit_ev = avg_excit_ev[0]
    #print(len(avg_excit_ev_err))
    #avg_excit_ev_err = np.mean(avg_excit_ev_err, axis = 0)
    avg_inhib_amount = np.mean(avg_inhib_amount, axis = 0) + np.mean(avg_td_amount, axis = 0)# - np.mean(avg_inhib_amount_tuned, axis = 0) - np.mean(avg_td_amount_tuned, axis =0)
    avg_inhib_amount_err = np.mean(avg_inhib_amount_err, axis = 0)
    
    arge = np.arange(1,start_space+1)
    diff = len(avg_spikes_noinhib) - start_space
    for value in range(1, diff+1):
        arge = np.append(arge, start_space + space * value)
    fig = plt.figure(1)
    ax = fig.add_axes([0,0,1,1])
    #p_ = ax.twinx()
    ax.set_xlabel("Length of oriented gratings (in pixels)")
    ax.set_ylabel("Normalized amount of spikes and inhibition")
    """avg_spikes_noinhib = ((avg_spikes_noinhib - avg_spikes_noinhib[0])/(max(avg_spikes_noinhib[0:12])))[0:12]
    avg_spikes_inhib = ((avg_spikes_inhib - avg_spikes_inhib[0])/(max(avg_spikes_inhib[0:12])))[0:12]
    avg_excit_ev = ((avg_excit_ev - avg_excit_ev[0])/(max(avg_excit_ev[0:12])))[0:12]
    avg_inhib_amount = ((avg_inhib_amount - avg_inhib_amount[0])/(max(avg_inhib_amount[0:12])))[0:12]"""

    avg_excit_ev = avg_excit_ev[0:12] 
    avg_spikes_noinhib = (avg_spikes_noinhib[0:12] - avg_spikes_noinhib[0])/(max(avg_spikes_noinhib)-avg_spikes_noinhib[0])
    avg_spikes_inhib = (avg_spikes_inhib[0:12] - avg_spikes_inhib[0])/(max(avg_spikes_inhib)-avg_spikes_inhib[0])
    avg_inhib_amount = (avg_inhib_amount[0:12] - avg_inhib_amount[0])/(avg_inhib_amount[12] - avg_inhib_amount[0])
    #ax.plot(arge[0:12], (avg_spikes_noinhib/max(avg_spikes_noinhib[0:12]))[0:12], 'k-', label = "spikes without inhibition")
    #ax.plot(arge[0:12], (avg_spikes_inhib/max(avg_spikes_inhib[0:12]))[0:12], 'r-', label = "spikes with inhibition")
    #ax.plot(arge[0:12], (avg_excit_ev), 'y-', label = "evolution of excitation")
    
    ax.plot(arge[0:12], (avg_spikes_inhib), 'g-', label = "spikes with inhibition")
    ax.plot(arge[0:12], (avg_spikes_noinhib), 'r-', label = "spikes without inhibition")
    #ax.plot(arge[0:12], (avg_inhib_amount), 'g-', label = "evolution of inhibition")
    
    steepness_a = (avg_inhib_amount[2] - avg_inhib_amount[0])/(3-1)
    ax.plot([arge[0], arge[2]], [avg_inhib_amount[0], avg_inhib_amount[2]], 'c-')#, label = "between 1 and 2 pixels")
    ax.plot([arge[0], arge[2]], [avg_inhib_amount[0], avg_inhib_amount[2]], 'go')
    steepness_b = (avg_inhib_amount[4] - avg_inhib_amount[2])/(5-3)
    ax.plot([arge[2], arge[4]], [avg_inhib_amount[2], avg_inhib_amount[4]], 'y-')#,label = "between 2 and 5 pixels")
    ax.plot([arge[2], arge[4]], [avg_inhib_amount[2], avg_inhib_amount[4]], 'go')
    steepness_c = (avg_inhib_amount[6] - avg_inhib_amount[4])/(7-5)
    ax.plot([arge[4], arge[6]], [avg_inhib_amount[4], avg_inhib_amount[6]], 'k-')#, label= "between 5 and 7 pixels")
    ax.plot([arge[4], arge[6]], [avg_inhib_amount[4], avg_inhib_amount[6]], 'go')
    steepness_d = (avg_inhib_amount[11] - avg_inhib_amount[6])/(12-7)
    ax.plot([arge[6], arge[11]], [avg_inhib_amount[6], avg_inhib_amount[11]], 'm-')#, label = "between 7 and 12 pixels")
    ax.plot([arge[6], arge[11]], [avg_inhib_amount[6], avg_inhib_amount[11]], 'go')

    print(steepness_a)
    print(steepness_b)
    print(steepness_c)
    print(steepness_d)
    
    """for val in range(len(arge)):
        ax.plot([arge[val], arge[val]], [avg_spikes_noinhib[val]-avg_spikes_noinhib_err[val],avg_spikes_noinhib[val]+avg_spikes_noinhib_err[val]], 'k--')
        ax.plot([arge[val], arge[val]], [avg_spikes_inhib[val]-avg_spikes_inhib_err[val],avg_spikes_inhib[val]+avg_spikes_inhib_err[val]], 'r--')
        #ax.plot([arge[val], arge[val]], [avg_excit_ev[val]-avg_excit_ev_err[val],avg_excit_ev[val]+avg_excit_ev_err[val]], 'y--')
        ax.plot([arge[val], arge[val]], [avg_inhib_amount[val]-avg_inhib_amount_err[val],avg_inhib_amount[val]+avg_inhib_amount_err[val]], 'g--')
    """
    plt.xticks(arge[0:12])
    fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))     
    
def compare_inhibitions(all_folder, td_folder, lat_folder, no_inhib_folder, thickness, start_space = 45, space = 3, n_simulation = 5):
    avg_spikes_noinhib = []
    avg_spikes_noinhib_err = []
    avg_spikes_inhib = []
    avg_spikes_inhib_err = []
    avg_spikes_lat = []
    avg_spikes_lat_err = []
    avg_spikes_td = []
    avg_spikes_td_err = []
    
    thickness_max = 3
    for thness in range(1,thickness_max+1):
        
        avg_spikes_noinhib.append(np.load(no_inhib_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_inhib.append(np.load(all_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_noinhib_err.append(np.load(no_inhib_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
        avg_spikes_inhib_err.append(np.load(all_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
        avg_spikes_lat.append(np.load(lat_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_td.append(np.load(td_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
        avg_spikes_lat_err.append(np.load(lat_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
        avg_spikes_td_err.append(np.load(td_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
       
    avg_spikes_noinhib = np.mean(avg_spikes_noinhib, axis = 0)
    avg_spikes_inhib = np.mean(avg_spikes_inhib, axis = 0)
    avg_spikes_noinhib_err = np.mean(avg_spikes_noinhib_err, axis = 0)
    avg_spikes_inhib_err = np.mean(avg_spikes_inhib_err, axis = 0)
    avg_spikes_lat = np.mean(avg_spikes_lat, axis = 0)
    avg_spikes_td = np.mean(avg_spikes_td, axis = 0)
    avg_spikes_lat_err = np.mean(avg_spikes_lat_err, axis = 0)
    avg_spikes_td_err = np.mean(avg_spikes_td_err, axis = 0)
    

    avg_spikes_noinhib = (avg_spikes_noinhib - avg_spikes_noinhib[0]) / (max(avg_spikes_noinhib) - avg_spikes_noinhib[0])
    avg_spikes_inhib = (avg_spikes_inhib - avg_spikes_inhib[0]) / (max(avg_spikes_inhib) - avg_spikes_inhib[0])
    avg_spikes_noinhib_err = (avg_spikes_noinhib_err- avg_spikes_noinhib_err[0]) / (max(avg_spikes_noinhib_err) - avg_spikes_noinhib_err[0])
    avg_spikes_inhib_err = (avg_spikes_inhib_err - avg_spikes_inhib_err[0]) / (max(avg_spikes_inhib_err) - avg_spikes_inhib_err[0])
    avg_spikes_lat = (avg_spikes_lat - avg_spikes_lat[0]) / (max(avg_spikes_lat) - avg_spikes_lat[0])
    avg_spikes_td = (avg_spikes_td - avg_spikes_td[0]) / (max(avg_spikes_td) - avg_spikes_td[0])
    avg_spikes_lat_err = (avg_spikes_lat_err - avg_spikes_lat_err[0]) / (max(avg_spikes_lat_err) - avg_spikes_lat_err[0])
    avg_spikes_td_err = (avg_spikes_td_err - avg_spikes_td_err[0]) / (max(avg_spikes_td_err) - avg_spikes_td_err[0])
    
    arge = np.arange(1,start_space+1)
    diff = len(avg_spikes_noinhib) - start_space
    for value in range(1, diff+1):
        arge = np.append(arge, start_space + space * value)
    fig = plt.figure(1)
    ax = fig.add_axes([0,0,1,1])
    #p_ = ax.twinx()
    ax.set_xlabel("Length of oriented gratings (in pixels)")
    ax.set_ylabel("Normalized amount of spikes")
    #ax.plot(arge[0:12], (avg_spikes_noinhib/max(avg_spikes_noinhib[0:12]))[0:12], 'k-', label = "spikes without inhibition")
    #ax.plot(arge[0:12], (avg_spikes_inhib/max(avg_spikes_inhib[0:12]))[0:12], 'r-', label = "spikes with inhibition")
    #ax.plot(arge[0:12], (avg_excit_ev), 'y-', label = "evolution of excitation")
    ax.plot(arge, avg_spikes_noinhib, 'k-', label = "no inhibition")
    ax.plot(arge, avg_spikes_td, 'b-', label = "top down inhibition")
    ax.plot(arge, avg_spikes_lat, 'y-', label = "lateral inhibition")
    ax.plot(arge, avg_spikes_inhib, 'r-', label = "all inhibition")
    
    

    ax.set_title("Comparison of the different inhibitions")

    
    """for val in range(len(arge)):
        ax.plot([arge[val], arge[val]], [avg_spikes_noinhib[val]-avg_spikes_noinhib_err[val],avg_spikes_noinhib[val]+avg_spikes_noinhib_err[val]], 'k--')
        ax.plot([arge[val], arge[val]], [avg_spikes_inhib[val]-avg_spikes_inhib_err[val],avg_spikes_inhib[val]+avg_spikes_inhib_err[val]], 'r--')
        #ax.plot([arge[val], arge[val]], [avg_excit_ev[val]-avg_excit_ev_err[val],avg_excit_ev[val]+avg_excit_ev_err[val]], 'y--')
        ax.plot([arge[val], arge[val]], [avg_inhib_amount[val]-avg_inhib_amount_err[val],avg_inhib_amount[val]+avg_inhib_amount_err[val]], 'g--')
    """
    plt.xticks(arge)
    fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))         

def orientations_graph(load_folder, thickness, angles, n_direction, start_space = 45, space = 3, n_simulation = 5):
    fig_number=0
    for angle in angles:
        for direction in range(n_direction):
            if(direction==0):
                str_angle = str(angle)
            else:
                str_angle = "-"+str(angle)
            fig_number+=1
            avg_spikes = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "spikes_avg.npy")
            avg_lat = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "lat_avg.npy")
            avg_td = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "td_avg.npy")
            std_spikes_err = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "spikes_stderr.npy")
            std_lat_err = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "lat_stderr.npy")
            std_td_err = np.load(load_folder+str(thickness)+"/"+str(angle)+"/"+str(direction)+"/" + "td_stderr.npy")
            arge = np.arange(1,start_space+1)
            diff = len(avg_spikes) - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(fig_number)
            ax = fig.add_axes([0,0,1,1])
            p_ = ax.twinx()
            ax.set_ylim([0, 1.7])
            p_.set_ylim([0, 1200])
            ax.set_xlabel("Length of oriented gratings (in pixels)")
            ax.set_ylabel("Average number of spikes")
            p_.set_ylabel("Evolution of total inhibition received by all neurons with the coordinates (4,4) in the neuronal map")
            ax.set_title("Evolution of spikes and inhibition of gratings of orientation " + str_angle + "° for " + str(n_simulation) + " simulations")
            p_.plot(arge, avg_lat, 'r-', label='lateral inhibition')
            p_.plot(arge, avg_td, 'g-', label='top-down inhibition')
            ax.plot(arge, avg_spikes, 'k-', label='spikes')
            for val in range(len(arge)):
                if(val==0):
                    p_.plot( [arge[val], arge[val]], [avg_lat[val]-std_lat_err[val],avg_lat[val]+std_lat_err[val]], 'r--',label='standard error of lateral inhibition')
                    p_.plot([arge[val], arge[val]], [avg_td[val]-std_td_err[val],avg_td[val]+std_td_err[val]], 'g--', label='standard error of top-down inhibition')
                    ax.plot([arge[val], arge[val]], [avg_spikes[val]-std_spikes_err[val],avg_spikes[val]+std_spikes_err[val]], 'k--', label='standard error of spikes')
                else:
                    p_.plot( [arge[val], arge[val]], [avg_lat[val]-std_lat_err[val],avg_lat[val]+std_lat_err[val]], 'r--')
                    p_.plot([arge[val], arge[val]], [avg_td[val]-std_td_err[val],avg_td[val]+std_td_err[val]], 'g--')
                    ax.plot([arge[val], arge[val]], [avg_spikes[val]-std_spikes_err[val],avg_spikes[val]+std_spikes_err[val]], 'k--')
            plt.xticks(arge)
            
            fig.legend(loc='center', bbox_to_anchor=(0.5, 0, 0.8, 0.35))
            
def calculate_similitude_tuning(load_folder, thickness, angles, n_direction):
    similitude_matrix = np.full((len(angles)*n_direction, len(angles)*n_direction),-5, dtype=float)
    print(similitude_matrix.shape)
    for index, angle in enumerate(angles):
        for direction in range(n_direction):
            str_angle = str(angle)+"/"+str(direction)
            for index_2, angle_2 in enumerate(angles):
                for direction_2 in range(n_direction):
                    str_angle_2 = str(angle_2)+"/"+str(direction_2)
                    if(similitude_matrix[direction*len(angles)+index][direction_2*len(angles)+index_2]!=float(-5)):
                        continue
                    else:
                        neurons = np.load(load_folder + str(thickness)+"/"+str_angle+"/" + "responsivecells_avg.npy")
                        neurons_2 = np.load(load_folder + str(thickness)+"/"+str_angle_2+"/" + "responsivecells_avg.npy")
                        complete = np.sort(np.unique(np.append(neurons, neurons_2)))
                        
                        index_new = 0
                        index_new_2 = 0
                        
                        for value in complete:
                            if(index_new<len(neurons)):
                                if(neurons[index_new]!=value):
                                    neurons = np.insert(neurons, index_new, -1)
                            else:
                                neurons = np.append(neurons,-1)
                            index_new+=1
                            if(index_new_2<len(neurons_2)):
                                if(neurons_2[index_new_2]!=value):
                                    neurons_2 = np.insert(neurons_2, index_new_2, -1)
                            else:
                                neurons_2 = np.append(neurons_2, -1)
                            index_new_2+=1
                        
                        corr = 1 - spatial.distance.cosine(neurons, neurons_2)
                        similitude_matrix[direction*len(angles)+index][direction_2*len(angles)+index_2] = corr
    return similitude_matrix
               
def average_orientations_numbers_cells(spinet: SpikingNetwork, n_thickness=3, depth = 64):
    avg_oris = np.zeros((len(spinet.neurons),n_thickness),dtype=float)
    counter_neurons = np.zeros((len(spinet.neurons),n_thickness),dtype=float)
    for index, layer in enumerate(spinet.neurons):
        for index2, neuron in enumerate(layer):
            if(index==0):
                if(index2==depth):
                    break
            
            for thickness in range(n_thickness):
                if(len(neuron.theta[thickness])!=1):
                    new_arr = np.unique(abs(np.array(neuron.theta[thickness])))
                    length = len(new_arr)
                    rx_0 = False
                    rx_180 = False
                    for value in new_arr:
                        if(value==0):
                            rx_0=True
                        elif(value==180):
                            rx_180=True
                        if(rx_0 and rx_180):
                            length=length-1
                            break
                    avg_oris[index][thickness]+=length
                    counter_neurons[index][thickness]+=1
                else:
                    if(neuron.theta[thickness][0]!=-1):
                        avg_oris[index][thickness]+=1
                        counter_neurons[index][thickness]+=1
    for i in range(len(spinet.neurons)):
        for j in range(n_thickness):
            avg_oris[i][j] /= counter_neurons[i][j]
    return avg_oris, counter_neurons

def calculate_similitude_preferred_orientation(spinet: SpikingNetwork, neuron_id=0, n_thickness=3, depth = 64):
    corr_neurons_simple = np.full((depth, n_thickness, n_thickness), -5, dtype=float)
    corr_neurons_complex = np.full((len(spinet.neurons[1]), n_thickness, n_thickness), -5, dtype=float)
    
    for i in range(depth):
        complete = np.array([])
        for thickness in range(n_thickness):
            complete = np.append(complete,spinet.neurons[0][neuron_id+i].theta[thickness])
        complete = np.sort(np.unique(abs(complete)))
        if(complete[-1]==180 and complete[0]==0):
            np.delete(complete, -1)
        elif(complete[-1]==180 and complete[0]!=0):
            complete = np.insert(complete, 0, 0)
            np.delete(complete, -1)
        index = np.zeros((n_thickness), dtype=int)
        neurons = [[]]
        for thickness in range(n_thickness):
            neurons[thickness]=spinet.neurons[0][neuron_id+i].theta[thickness]
            neurons[thickness]=np.sort(np.unique(abs(np.array(neurons[thickness]))))
            if(neurons[thickness][-1]==180 and neurons[thickness][0]==0):
                np.delete(neurons[thickness], -1)
            elif(neurons[thickness][-1]==180 and neurons[thickness][0]!=0):
                neurons[thickness] = np.insert(neurons[thickness], 0, 0)
                np.delete(neurons[thickness], -1)
            for value in complete:
                if(index[thickness]<len(neurons[thickness])):
                    if(neurons[thickness][index[thickness]]!=value):
                        neurons[thickness] = np.insert(neurons[thickness], index[thickness], -1)
                else:
                    neurons[thickness] = np.append(neurons[thickness],-1)
                index[thickness]+=1
            if(thickness!=n_thickness-1):
               neurons.append([])     
        for thickness in range(n_thickness):
            for thick in range(n_thickness):
                if(corr_neurons_simple[i][thickness][thick]==float(-5)):
                    corr = 1 - spatial.distance.cosine(neurons[thickness], neurons[thick])
                    corr_neurons_simple[i][thickness][thick] = corr
    
    for j in range(len(spinet.neurons[1])):
        complete = np.array([])
        for thickness in range(n_thickness):
            complete = np.append(complete, spinet.neurons[1][j].theta[thickness])
        complete = np.sort(np.unique(abs(complete)))
        if(complete[-1]==180 and complete[0]==0):
            np.delete(complete, -1)
        elif(complete[-1]==180 and complete[0]!=0):
            complete = np.insert(complete, 0, 0)
            np.delete(complete, -1)
        
        index = np.zeros((n_thickness), dtype=int)
        neurons = [[]]
        for thickness in range(n_thickness):
            neurons[thickness]=spinet.neurons[1][j].theta[thickness]
            neurons[thickness]=np.sort(np.unique(abs(np.array(neurons[thickness]))))
            if(neurons[thickness][-1]==180 and neurons[thickness][0]==0):
                np.delete(neurons[thickness], -1)
            elif(neurons[thickness][-1]==180 and neurons[thickness][0]!=0):
                neurons[thickness] = np.insert(neurons[thickness], 0, 0)
                np.delete(neurons[thickness], -1)
            for value in complete:
                if(index[thickness]<len(neurons[thickness])):
                    if(neurons[thickness][index[thickness]]!=value):
                        neurons[thickness] = np.insert(neurons[thickness], index[thickness], -1)
                else:
                    neurons[thickness] = np.append(neurons[thickness],-1)
                index[thickness]+=1
            if(thickness!=n_thickness-1):
               neurons.append([])     
        for thickness in range(n_thickness):
            for thick in range(n_thickness):
                if(corr_neurons_complex[j][thickness][thick]==float(-5)):
                    corr = 1 - spatial.distance.cosine(neurons[thickness], neurons[thick])
                    corr_neurons_complex[j][thickness][thick] = corr
                    
    return np.mean(corr_neurons_simple,axis=0), np.mean(corr_neurons_complex, axis=0)

def count_orientations_per_thickness(spinet: SpikingNetwork, neuron_id=0, n_thickness=1, depth=64, angles=[0, 23, 45, 68, 90, 113, 135, 158]):
    counter_angles_simple = np.zeros((len(angles), n_thickness), dtype=int)
    counter_angles_complex = np.zeros((len(angles), n_thickness), dtype=int)
    for i in range(depth):
        for thickness in range(n_thickness):
            unique_vals = np.sort(np.unique(abs(np.array(spinet.neurons[0][neuron_id+i].theta[thickness]))))
            if(unique_vals[-1]==180 and unique_vals[0]==0):
                np.delete(unique_vals, -1)
            elif(unique_vals[-1]==180 and unique_vals[0]!=0):
                unique_vals = np.insert(unique_vals, 0, 0)
                np.delete(unique_vals, -1)
            for index, angle in enumerate(angles):
                if((unique_vals==angle).any()):
                    counter_angles_simple[index][thickness]+=1
                    
    for i in range(len(spinet.neurons[1])):
        for thickness in range(n_thickness):
            unique_vals = np.sort(np.unique(abs(np.array(spinet.neurons[1][neuron_id+i].theta[thickness]))))
            if(unique_vals[-1]==180 and unique_vals[0]==0):
                np.delete(unique_vals, -1)
            elif(unique_vals[-1]==180 and unique_vals[0]!=0):
                unique_vals = np.insert(unique_vals, 0, 0)
                np.delete(unique_vals, -1)
            for index, angle in enumerate(angles):
                if((unique_vals==angle).any()):
                    counter_angles_complex[index][thickness]+=1
    
    return np.mean(counter_angles_simple,axis=1), np.mean(counter_angles_complex,axis=1)
   
def direction_invariance(spinet: SpikingNetwork, neuron_id=0, n_thickness=3, depth=64):
    direction_simple = np.zeros((depth))
    direction_complex = np.zeros((len(spinet.neurons[1])))
    for i in range(depth):
        complete_normal = np.array([])
        for thickness in range(n_thickness):
            complete_normal = np.append(complete_normal,spinet.neurons[0][neuron_id+i].theta[thickness])
        complete = np.sort(np.unique(abs(complete_normal)))
        if(complete[-1]==180 and complete[0]==0):
            np.delete(complete, -1)
        elif(complete[-1]==180 and complete[0]!=0):
            complete = np.insert(complete, 0, 0)
            np.delete(complete, -1)
        for value in complete:
            if(value==0):
                if((complete_normal==180).any()):
                    direction_simple[i]+=1
            else:
                if((complete_normal==-value).any()):
                    direction_simple[i]+=1
        direction_simple[i]/=len(complete)
        
    for i in range(len(spinet.neurons[1])):
        complete_normal = np.array([])
        for thickness in range(n_thickness):
            complete_normal = np.append(complete_normal,spinet.neurons[1][neuron_id+i].theta[thickness])
        complete = np.sort(np.unique(abs(complete_normal)))
        if(complete[-1]==180 and complete[0]==0):
            np.delete(complete, -1)
        elif(complete[-1]==180 and complete[0]!=0):
            complete = np.insert(complete, 0, 0)
            np.delete(complete, -1)
        for value in complete:
            if(value==0):
                if((complete_normal==180).any()):
                    direction_complex[i]+=1
            else:
                if((complete_normal==-value).any()):
                    direction_complex[i]+=1
        direction_complex[i]/=len(complete)
    return direction_simple, direction_complex


###########################################SCRIPT AVERAGED GRAPH; OBSOLETE############################################

def averaged_graph(load_folder, thickness, start_space = 45, space = 3, n_simulation = 5, average_all = False, load_folder2 = "full", see_inhib = False, with_speed = False, speed = 0):
    if(not average_all and not see_inhib):
        if(not with_speed):
            avg_spikes = np.load(load_folder+str(thickness)+"/orientations_average/" + "spikes_avg.npy")
    
            avg_lat = np.load(load_folder+str(thickness)+"/orientations_average/" + "lattuned_avg.npy")
            avg_td = np.load(load_folder+str(thickness)+"/orientations_average/" + "tdtuned_avg.npy")
            std_spikes_err = np.load(load_folder+str(thickness)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation)
    
            std_lat_err = np.load(load_folder+str(thickness)+"/orientations_average/" + "lattuned_stderr.npy")
            std_td_err = np.load(load_folder+str(thickness)+"/orientations_average/" + "tdtuned_stderr.npy") 
            
            
            """avg_spikes2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "spikes_avg.npy")
    
            avg_lat2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "lattuned_avg.npy")
            avg_td2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "tdtuned_avg.npy")
            std_spikes_err2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation)
    
            std_lat_err2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "lattuned_stderr.npy")
            std_td_err2 = np.load(load_folder+str(thickness+1)+"/orientations_average/" + "tdtuned_stderr.npy") 
            
            
            
            avg_spikes3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "spikes_avg.npy")
    
            avg_lat3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "lattuned_avg.npy")
            avg_td3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "tdtuned_avg.npy")
            std_spikes_err3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation)
    
            std_lat_err3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "lattuned_stderr.npy")
            std_td_err3 = np.load(load_folder+str(thickness+2)+"/orientations_average/" + "tdtuned_stderr.npy") """
            
            
        else:
            avg_spikes = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) + "/orientations_average/" + "spikes_avg.npy")
    
            avg_lat = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_avg.npy")
            avg_td = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "tdtuned_avg.npy")
            std_spikes_err = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation)
    
            std_lat_err = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_stderr.npy")
            std_td_err = np.load(load_folder+str(thickness)+"/speeds/" + str(speed) +"/orientations_average/" + "tdtuned_stderr.npy") 
        arge = np.arange(1,start_space+1)
        diff = len(avg_spikes) - start_space
        for value in range(1, diff+1):
            arge = np.append(arge, start_space + space * value)
        fig = plt.figure(1)
        ax = fig.add_axes([0,0,1,1])
        #p_ = ax.twinx()
        ax.set_xlabel("Length of oriented gratings (in pixels)")
        ax.set_ylabel("Average number of spikes")
        #p_.set_ylabel("Average amount of inhibition received by a neuron")
        #ax.set_title("Evolution of spikes and inhibition over all orientations for " + str(n_simulation) + " simulations for a thickness of " + str(thickness) + " for simple cells at the center of the neuronal map")
        #ax.set_title("Evolution of spikes without any inhibition over all orientations for " + str(n_simulation) + " simulations for a thickness of " + str(thickness) + " for simple cells at the center of the neuronal map")
        
        ax.set_title("Evolution of spikes for gratings of different thickness")
        #p_.plot(arge, avg_lat, 'r-', label='lateral inhibition')
        #p_.plot(arge, avg_td, 'g-', label='top-down inhibition')
        print(avg_spikes)
        ax.plot(arge, avg_spikes, 'b-', label='spikes for thickness = 1')
        """ax.plot(arge, avg_spikes2, 'g-', label='spikes for thickness = 2')
        ax.plot(arge, avg_spikes3, 'r-', label='spikes for thickness = 3')"""

        
        """for val in range(len(arge)):
            if(val==0):
                #p_.plot( [arge[val], arge[val]], [avg_lat[val]-std_lat_err[val],avg_lat[val]+std_lat_err[val]], 'r--',label='standard error of lateral inhibition')
                #p_.plot([arge[val], arge[val]], [avg_td[val]-std_td_err[val],avg_td[val]+std_td_err[val]], 'g--', label='standard error of top-down inhibition')
                ax.plot([arge[val], arge[val]], [avg_spikes[val]-std_spikes_err[val],avg_spikes[val]+std_spikes_err[val]], 'b-', label='standard error of spikes for thickness = 1')
                ax.plot([arge[val], arge[val]], [avg_spikes2[val]-std_spikes_err2[val],avg_spikes2[val]+std_spikes_err2[val]], 'g--', label='standard error of spikes for thickness = 2')
                ax.plot([arge[val], arge[val]], [avg_spikes3[val]-std_spikes_err3[val],avg_spikes3[val]+std_spikes_err3[val]], 'r--', label='standard error of spikes for thickness = 3')

            else:
                #p_.plot( [arge[val], arge[val]], [avg_lat[val]-std_lat_err[val],avg_lat[val]+std_lat_err[val]], 'r--')
                #p_.plot([arge[val], arge[val]], [avg_td[val]-std_td_err[val],avg_td[val]+std_td_err[val]], 'g--')
                ax.plot([arge[val], arge[val]], [avg_spikes[val]-std_spikes_err[val],avg_spikes[val]+std_spikes_err[val]], 'k--')
                ax.plot([arge[val], arge[val]], [avg_spikes2[val]-std_spikes_err2[val],avg_spikes2[val]+std_spikes_err2[val]], 'g--')
                ax.plot([arge[val], arge[val]], [avg_spikes3[val]-std_spikes_err3[val],avg_spikes3[val]+std_spikes_err3[val]], 'r--')
        """
        suppr_percentage = suppression_metric(avg_spikes)
        ax.text(31, max(avg_spikes)/2, "Suppression up to " + str(int(suppr_percentage)) + "%", fontsize = 42)
        plt.xticks(arge)
        fig.legend(loc='center', bbox_to_anchor=(0.5, 0, 0.8, 0.35))
        return avg_spikes
    elif(not see_inhib):
        avg_thness_spikes = []
        avg_thness_lat = []
        avg_thness_td = []
        avg_thness_stattuned = []
        avg_thness_stat = []

        avg_thness_std_spikes_err = []
        avg_thness_std_lat_err = []
        avg_thness_std_td_err = []
        
        avg_thness_spikes2 = []
        avg_thness_lat2 = []
        avg_thness_td2 = []
        avg_thness_tdtuned2 = []
        avg_thness_lattuned2 = []
        avg_thness_stattuned2 = []
        avg_thness_stat2 = []
        
        avg_thness_std_spikes_err2 = []
        avg_thness_std_lat_err2 = []
        avg_thness_std_td_err2 = []
        
        avg_thness_spikes3= []
        avg_thness_lat3 = []
        avg_thness_td3 = []
        avg_thness_lattuned3 = []
        avg_thness_stattuned3 = []
        avg_thness_stat3 = []
        
        avg_thness_std_spikes_err3 = []
        avg_thness_std_lat_err3 = []
        avg_thness_std_td_err3 = []
        
        avg_thness_spikes4 = []
        avg_thness_spikes5 = []
        
        avg_thness_spikes6 = []
        avg_thness_spikes7 = []
        avg_thness_spikes8 = []
        
        avg_thness_lat4 = []
        avg_thness_td4 = []
        avg_thness_lattuned4 = []
        avg_thness_stattuned4 = []
        avg_thness_stat4 = []
        
        avg_thness_std_spikes_err4 = []
        avg_thness_std_lat_err4 = []
        avg_thness_std_td_err4 = []
        
        avg_thness_spikes5 = []
        avg_thness_lat5 = []
        avg_thness_td5 = []
        avg_thness_tdtuned5 = []
        avg_thness_lattuned5 = []
        avg_thness_stattuned5 = []
        avg_thness_stat5 = []
        
        avg_thness_std_spikes_err5 = []
        avg_thness_std_lat_err5 = []
        avg_thness_std_td_err5 = []
        
        avg_thness_spikes6= []
        avg_thness_lat6 = []
        avg_thness_td6 = []
        avg_thness_lattuned6 = []
        avg_thness_stattuned6 = []
        avg_thness_stat6 = []
        
        avg_thness_std_spikes_err6 = []
        avg_thness_std_lat_err6 = []
        avg_thness_std_td_err6 = []
        
        load_folder3="/home/comsee/Internship_Antony/neuvisys/save_files/oris/topdownsolo/"
        #load_folder3 = "/home/comsee/Internship_Antony/neuvisys/save_files/oris/topdownshuffled/"
        load_folder4= "/home/comsee/Internship_Antony/neuvisys/save_files/oris/latsolo/"
        load_folder6= "/home/comsee/Internship_Antony/neuvisys/save_files/oris/with_speeds_nonsynth/"

        thickness_max = 3
        for thness in range(1,thickness_max+1):
            
            avg_thness_spikes.append(np.load(load_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
            avg_thness_lat.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "lat_avg.npy"))
            avg_thness_td.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "td_avg.npy"))
            avg_thness_stat.append(np.load(load_folder+str(thness)+"/orientations_average/"+"stat_avg.npy"))

            avg_thness_std_spikes_err.append(np.load(load_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
            avg_thness_std_lat_err.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
            avg_thness_std_td_err.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
            
            if(not with_speed):
                avg_thness_spikes2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
                avg_thness_lat2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                avg_thness_td2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "td_avg.npy"))
                avg_thness_tdtuned2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "tdtuned_avg.npy"))
                avg_thness_stattuned2.append(np.load(load_folder2+str(thness)+"/orientations_average/"+"stattuned_avg.npy"))
                avg_thness_stat2.append(np.load(load_folder2+str(thness)+"/orientations_average/"+"stat_avg.npy"))
                avg_thness_lattuned2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lattuned_avg.npy"))

    
                avg_thness_std_spikes_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_lat_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_td_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                
                avg_thness_spikes3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
                avg_thness_lat3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                avg_thness_td3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "td_avg.npy"))
                avg_thness_stat3.append(np.load(load_folder3+str(thness)+"/orientations_average/"+"stat_avg.npy"))
                avg_thness_lattuned3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "lattuned_avg.npy"))

    
                avg_thness_std_spikes_err3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_lat_err3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_td_err3.append(np.load(load_folder3+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                
                avg_thness_spikes4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
                avg_thness_lat4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                avg_thness_td4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "td_avg.npy"))
                avg_thness_stat4.append(np.load(load_folder4+str(thness)+"/orientations_average/"+"stat_avg.npy"))

    
                avg_thness_std_spikes_err4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_lat_err4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                avg_thness_std_td_err4.append(np.load(load_folder4+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                
            else:
                #avg_thness_excit_ev.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "events_avg.npy"))
                if(thness<=3):
                    avg_thness_spikes2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_thness_lat2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_avg.npy"))
                    avg_thness_td2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_avg.npy"))
                    avg_thness_stattuned2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/"+"stattuned_avg.npy"))
                    avg_thness_stat2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/"+"stat_avg.npy"))
                    avg_thness_lattuned2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_avg.npy"))
    
        
                    avg_thness_std_spikes_err2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                    avg_thness_std_lat_err2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_stderr.npy"))
                    avg_thness_std_td_err2.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_stderr.npy"))
                    
                    #print(speed+1)
                    avg_thness_spikes3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_thness_lat3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lat_avg.npy"))
                    avg_thness_td3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "td_avg.npy"))
                    avg_thness_stattuned3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/"+"stattuned_avg.npy"))
                    avg_thness_stat3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/"+"stat_avg.npy"))
                    avg_thness_lattuned3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lattuned_avg.npy"))
    
        
                    avg_thness_std_spikes_err3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                    avg_thness_std_lat_err3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lat_stderr.npy"))
                    avg_thness_std_td_err3.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "td_stderr.npy"))
                    
                    avg_thness_spikes4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_thness_lat4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "lat_avg.npy"))
                    avg_thness_td4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "td_avg.npy"))
                    avg_thness_stattuned4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/"+"stattuned_avg.npy"))
                    avg_thness_stat4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/"+"stat_avg.npy"))
                    avg_thness_lattuned4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "lattuned_avg.npy"))
    
        
                    avg_thness_std_spikes_err4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation))
                    avg_thness_std_lat_err4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "lat_stderr.npy"))
                    avg_thness_std_td_err4.append(np.load(load_folder2+str(thness)+"/speeds/" + str(speed+2) +"/orientations_average/" + "td_stderr.npy"))
                    
                if(thness <3):
                    avg_thness_spikes5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_thness_lat5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_avg.npy"))
                    avg_thness_td5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_avg.npy"))
                    avg_thness_stattuned5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/"+"stattuned_avg.npy"))
                    avg_thness_stat5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/"+"stat_avg.npy"))
                    avg_thness_lattuned5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lattuned_avg.npy"))
    
        
                    avg_thness_std_spikes_err5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation-1))
                    avg_thness_std_lat_err5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "lat_stderr.npy"))
                    avg_thness_std_td_err5.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed) +"/orientations_average/" + "td_stderr.npy"))
                    
                    #print(speed+1)
                    avg_thness_spikes6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_thness_lat6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lat_avg.npy"))
                    avg_thness_td6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "td_avg.npy"))
                    avg_thness_stattuned6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/"+"stattuned_avg.npy"))
                    avg_thness_stat6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/"+"stat_avg.npy"))
                    avg_thness_lattuned6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lattuned_avg.npy"))
    
        
                    avg_thness_std_spikes_err6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "spikes_stderr.npy")* np.sqrt(n_simulation-1))
                    avg_thness_std_lat_err6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "lat_stderr.npy"))
                    avg_thness_std_td_err6.append(np.load(load_folder6+str(thness)+"/speeds/" + str(speed+1) +"/orientations_average/" + "td_stderr.npy"))
                    
            
        arge = np.arange(1,start_space+1)
        diff = len(avg_thness_spikes[0]) - start_space
        for value in range(1, diff+1):
            arge = np.append(arge, start_space + space * value)
        fig = plt.figure(1)
        ax = fig.add_axes([0,0,1,1])
        #p_ = ax.twinx()
        ax.set_xlabel("Length of oriented gratings (in pixels)")
        ax.set_ylabel("Average number of spikes")
        """p_.spines['left'].set_color('red')
        p_.spines['left'].set_linewidth(3)
        ax.tick_params(axis = 'y', colors='red', which='major')
        p_.spines['right'].set_linewidth(3)
        p_.spines['right'].set_color('green')
        p_.tick_params(axis = 'y', colors='green', which='major')
        p_.set_ylabel("Average amount of inhibition per cell")"""
        #ax.set_title("Evolution of spikes and inhibition over all orientations for " + str(n_simulation) + "  simulations for all thicknesses (1 to 3) for simple cells at the center of the neuronal map")
        #ax.set_title("Comparison of network's activity with and without inhibition averaged on all orientations for simple cells according to their orientation's preference")
        #ax.set_title("Comparison of network's activity with different top down inhibition scaled")
        ax.set_title("Comparison of average number of spikes with different inhibitions")


        #print(len(avg_lat))
        
        avg_thness_spikes = np.mean(avg_thness_spikes,axis=0)
        avg_thness_lat = np.mean(avg_thness_lat, axis=0)
        avg_thness_td = np.mean(avg_thness_td, axis=0)
        #avg_thness_stattuned = np.mean(avg_thness_stattuned, axis=0)
        avg_thness_stat = np.mean(avg_thness_stat, axis=0)
        avg_thness_std_spikes_err = np.mean(avg_thness_std_spikes_err,axis=0)
        avg_thness_std_lat_err = np.mean(avg_thness_std_lat_err, axis=0)
        avg_thness_std_td_err = np.mean(avg_thness_std_td_err, axis=0)
        
        #print(avg_thness_stat2)

        avg_thness_spikes2 = np.mean(avg_thness_spikes2,axis=0)
        avg_thness_lat2 = np.mean(avg_thness_lat2, axis=0)
        avg_thness_td2 = np.mean(avg_thness_td2, axis=0)
        #avg_thness_tdtuned2 = np.mean(avg_thness_tdtuned2, axis=0)
        #avg_thness_stattuned2 = np.mean(avg_thness_stattuned2, axis=0)
        avg_thness_stat2 = np.mean(avg_thness_stat2, axis=0)
        #avg_thness_lattuned2 = np.mean(avg_thness_lattuned2, axis=0)
        avg_thness_std_spikes_err2 = np.mean(avg_thness_std_spikes_err2,axis=0)
        avg_thness_std_lat_err2 = np.mean(avg_thness_std_lat_err2, axis=0)
        avg_thness_std_td_err2 = np.mean(avg_thness_std_td_err2, axis=0)
        
        avg_thness_spikes3 = np.mean(avg_thness_spikes3,axis=0)
        avg_thness_lat3 = np.mean(avg_thness_lat3, axis=0)
        avg_thness_td3 = np.mean(avg_thness_td3, axis=0)
        avg_thness_stat3 = np.mean(avg_thness_stat3, axis=0)
        avg_thness_std_spikes_err3 = np.mean(avg_thness_std_spikes_err3,axis=0)
        avg_thness_std_lat_err3 = np.mean(avg_thness_std_lat_err3, axis=0)
        avg_thness_std_td_err3 = np.mean(avg_thness_std_td_err3, axis=0)
        
        avg_thness_spikes4 = np.mean(avg_thness_spikes4,axis=0)
        avg_thness_lat4 = np.mean(avg_thness_lat4, axis=0)
        avg_thness_td4 = np.mean(avg_thness_td4, axis=0)
        avg_thness_stat4 = np.mean(avg_thness_stat4, axis=0)
        avg_thness_std_spikes_err4 = np.mean(avg_thness_std_spikes_err4,axis=0)
        avg_thness_std_lat_err4 = np.mean(avg_thness_std_lat_err4, axis=0)
        avg_thness_std_td_err4 = np.mean(avg_thness_std_td_err4, axis=0)

        """avg_thness_spikes5 = np.mean(avg_thness_spikes5,axis=0)
        avg_thness_lat5 = np.mean(avg_thness_lat5, axis=0)
        avg_thness_td5 = np.mean(avg_thness_td5, axis=0)
        avg_thness_stat5 = np.mean(avg_thness_stat5, axis=0)
        avg_thness_std_spikes_err5 = np.mean(avg_thness_std_spikes_err5,axis=0)
        avg_thness_std_lat_err5 = np.mean(avg_thness_std_lat_err5, axis=0)
        avg_thness_std_td_err5 = np.mean(avg_thness_std_td_err5, axis=0)
        
        avg_thness_spikes6 = np.mean(avg_thness_spikes6,axis=0)
        avg_thness_lat6 = np.mean(avg_thness_lat6, axis=0)
        avg_thness_td6 = np.mean(avg_thness_td6, axis=0)
        avg_thness_stat6 = np.mean(avg_thness_stat6, axis=0)
        avg_thness_std_spikes_err6 = np.mean(avg_thness_std_spikes_err6,axis=0)
        avg_thness_std_lat_err6 = np.mean(avg_thness_std_lat_err6, axis=0)
        avg_thness_std_td_err6 = np.mean(avg_thness_std_td_err6, axis=0)"""
        
        #avg_thness_lat = (avg_thness_lat-min(avg_thness_lat))/(max(avg_thness_lat)-min(avg_thness_lat))
        #avg_thness_td = (avg_thness_td-min(avg_thness_td))/(max(avg_thness_td)-min(avg_thness_td))

        #p_.plot(arge, avg_thness_lat, 'g-', label='lateral inhibition')
        #p_.plot(arge, avg_thness_td, 'g-', label='top down inhibition')

        #p_.plot(arge, avg_thness_lat2, 'r-', label='lateral inhibition')
        #ax.plot(arge, (avg_thness_spikes), 'k-', label="without inhibition")
        """avg= []
        avg.append(avg_thness_spikes2)
        avg.append(avg_thness_spikes3)
        avg.append(avg_thness_spikes4)
        avg = np.mean(avg,axis = 0)
        
        avg_err= []
        avg_err.append(avg_thness_std_spikes_err2)
        avg_err.append(avg_thness_std_spikes_err3)
        avg_err.append(avg_thness_std_spikes_err4)
        avg_err = np.mean(avg_err, axis=0)
        
        avgreal= []
        avgreal.append(avg_thness_spikes5)
        avgreal.append(avg_thness_spikes6)
        avgreal = np.mean(avgreal,axis = 0)
        
        avg_errreal= []
        avg_errreal.append(avg_thness_std_spikes_err5)
        avg_errreal.append(avg_thness_std_spikes_err6)
        avg_errreal = np.mean(avg_errreal, axis=0)"""
        
        ax.plot(arge, avg_thness_spikes, 'r-', label="no dynamic inhibition")
        #p_.plot(arge, avgreal, 'b-', label="natural events")
        #ax.set_ylim([0, 0.405])

        ax.plot(arge, avg_thness_spikes4, 'b-', label="lateral dynamic inhibition")
        ax.plot(arge, avg_thness_spikes3, 'g-', label="top down dynamic inhibition")
        ax.plot(arge, avg_thness_spikes2, 'k-', label="all dynamic inhibitions")
        
        

        #p_.plot(arge, avg_thness_spikes2, 'b-', label="real life data")

        #ax.plot(arge, avg_thness_spikes3, 'g-', label="spikes with medium speed")

        #ax.plot(arge, avg_thness_spikes4, 'r-', label="spikes with high speed")

        #p_.plot(arge, (avg_thness_spikes2), 'r-', label="with inhibition")
        #p_.plot(arge[0:12], avg_thness_td2[0:12] - avg_thness_tdtuned2[0:12] + avg_thness_lat2[0:12] - avg_thness_lattuned2[0:12])
        #p_.plot(arge,avg_thness_lattuned2)
        #ax.plot(arge, avg_thness_stattuned, 'k--', label="static inhibition without dynamic")
        #p_.plot(arge, avg_thness_lat2 , 'r--', label="static inhibition with dynamic")
        
        for val in range(len(arge)):
            #p_.plot( [arge[val], arge[val]], [avg_thness_lat[val]-avg_thness_std_lat_err[val],avg_thness_lat[val]+avg_thness_std_lat_err[val]], 'g--')
            #p_.plot([arge[val], arge[val]], [avg_thness_td[val]-avg_thness_std_td_err[val],avg_thness_td[val]+avg_thness_std_td_err[val]], 'g--')
            ax.plot([arge[val], arge[val]], [avg_thness_spikes[val]-avg_thness_std_spikes_err[val],avg_thness_spikes[val]+avg_thness_std_spikes_err[val]], 'r--')
    
            #p_.plot( [arge[val], arge[val]], [avg_thness_lat2[val]-avg_thness_std_lat_err2[val],avg_thness_lat2[val]+avg_thness_std_lat_err2[val]], 'r--')
            #p_.plot([arge[val], arge[val]], [avg_thness_td2[val]-avg_thness_std_td_err2[val],avg_thness_td2[val]+avg_thness_std_td_err2[val]], 'g--')
            ax.plot([arge[val], arge[val]], [avg_thness_spikes2[val]-avg_thness_std_spikes_err2[val],avg_thness_spikes2[val]+avg_thness_std_spikes_err2[val]], 'k--')
            
            ax.plot([arge[val], arge[val]], [avg_thness_spikes3[val]-avg_thness_std_spikes_err3[val],avg_thness_spikes3[val]+avg_thness_std_spikes_err3[val]], 'g--')
            ax.plot([arge[val], arge[val]], [avg_thness_spikes4[val]-avg_thness_std_spikes_err4[val],avg_thness_spikes4[val]+avg_thness_std_spikes_err4[val]], 'b--')
            #ax.plot([arge[val], arge[val]], [avg[val]-avg_err[val],avg[val]+avg_err[val]], 'r--')
            #p_.plot([arge[val], arge[val]], [avgreal[val]-avg_errreal[val],avgreal[val]+avg_errreal[val]], 'b--')
        #plt.axvline(x = 7, color = 'k', linestyle = 'dashed', label="overlap")    
        #plt.axvline(x = 12, color = 'k', linestyle = 'dashed')    
        
        """suppr_percentage2 = suppression_metric(avg_thness_spikes2)
        suppr_percentage3 = suppression_metric(avg_thness_spikes3)
        suppr_percentage4 = suppression_metric(avg_thness_spikes4)"""
        
        suppr_percentage = suppression_metric(avg_thness_spikes)
        print(suppr_percentage)
        """suppr_percentage = suppression_metric(avgreal)
        print(suppr_percentage)
        
        print(suppr_percentage2)
        print(suppr_percentage3)
        print(suppr_percentage4)"""

        #ax.text(6, 0.03, "Synthetic suppression of " + "{:.2f}".format(suppr_percentage2) + "%" + " and real life data suppression of " + "{:.2f}".format(suppr_percentage) + "%" , fontsize = 42)

        #ax.text(31, 1, "Difference of " + str(int(suppr_percentage2 - suppr_percentage)) + "%", fontsize = 42)
        #ax.text(31, 1, "Difference of " + str((mean_0 - mean_1)) + "%", fontsize = 42)

        plt.xticks(arge)
        fig.legend(loc='center', bbox_to_anchor=(0.5, 0, 0.7, 0.25))
        return avg_thness_spikes, avg_thness_std_spikes_err, arge
    elif(see_inhib):
        avg_thness_spikes = []
        avg_thness_lat = []
        avg_thness_td = []
        avg_thness_std_spikes_err = []
        avg_thness_std_lat_err = []
        avg_thness_std_td_err = []
        
        avg_thness_spikes2 = []
        avg_thness_lat2 = []
        avg_thness_td2 = []
        avg_thness_std_spikes_err2 = []
        avg_thness_std_lat_err2 = []
        avg_thness_std_td_err2 = []
        avg_thness_lat2_all = []
        avg_thness_td2_all = []
        
        thickness_max = 3
        for thness in range(1,thickness_max+1):
            avg_thness_spikes.append(np.load(load_folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
            """avg_thness_lat.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "lattuned_avg.npy"))
            avg_thness_td.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "tdtuned_avg.npy"))"""
            avg_thness_std_spikes_err.append(np.load(load_folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
            """avg_thness_std_lat_err.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "lattuned_stderr.npy"))
            avg_thness_std_td_err.append(np.load(load_folder+str(thickness)+"/orientations_average/" + "tdtuned_stderr.npy"))"""
            
            avg_thness_spikes2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
            avg_thness_lat2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lattuned_avg.npy"))
            avg_thness_td2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "tdtuned_avg.npy"))
            avg_thness_std_spikes_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "spikes_stderr.npy"))
            avg_thness_std_lat_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lattuned_stderr.npy"))
            avg_thness_std_td_err2.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "td_stderr.npy"))
            avg_thness_lat2_all.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "lat_avg.npy"))
            avg_thness_td2_all.append(np.load(load_folder2+str(thness)+"/orientations_average/" + "td_avg.npy"))
        arge = np.arange(1,start_space+1)
        diff = len(avg_thness_spikes[0]) - start_space
        for value in range(1, diff+1):
            arge = np.append(arge, start_space + space * value)
        fig = plt.figure(1)
        ax = fig.add_axes([0,0,1,1])
        #p_ = ax.twinx()
        ax.set_xlabel("Length of oriented gratings (in pixels)")
        ax.set_ylabel("Average amount of inhibition received by a cell")
        #p_.set_ylabel("Average number of spikes (with inhibition)")
        #ax.set_title("Evolution of spikes and inhibition over all orientations for " + str(n_simulation) + "  simulations for all thicknesses (1 to 3) for simple cells at the center of the neuronal map")
        ax.set_title("Comparison of average amount of inhibition received by simple cells of correct orientation preference and others (for lateral and topdown)")
        #print(len(avg_lat))
        
        avg_thness_spikes = np.mean(avg_thness_spikes,axis=0)
        """avg_thness_lat = np.mean(avg_thness_lat)
        avg_thness_td = np.mean(avg_thness_td)"""
        avg_thness_std_spikes_err = np.mean(avg_thness_std_spikes_err,axis=0)
        """avg_thness_std_lat_err = np.mean(avg_thness_std_lat_err)
        avg_thness_std_td_err = np.mean(avg_thness_std_td_err)"""
        
        avg_thness_spikes2 = np.mean(avg_thness_spikes2,axis=0)
        avg_thness_lat2 = np.mean(avg_thness_lat2, axis=0)
        avg_thness_lat2_all = np.mean(avg_thness_lat2_all, axis=0)
        avg_thness_td2_all = np.mean(avg_thness_td2_all, axis=0)

        avg_thness_td2 = np.mean(avg_thness_td2, axis=0)
        avg_thness_std_spikes_err2 = np.mean(avg_thness_std_spikes_err2,axis=0)
        avg_thness_std_lat_err2 = np.mean(avg_thness_std_lat_err2, axis = 0)
        avg_thness_std_td_err2 = np.mean(avg_thness_std_td_err2, axis = 0)
        
        ax.plot(arge, avg_thness_td2, 'g-', label='topdown inhibition received by simple cells of orientation preference')
        ax.plot(arge, avg_thness_lat2, 'r-', label='lateral inhibition received by simple cells of orientation preference')
        
        ax.plot(arge, avg_thness_td2_all-avg_thness_td2, 'g--', label='topdown inhibition received by other simple cells')
        ax.plot(arge, avg_thness_lat2_all-avg_thness_lat2, 'r--', label='lateral inhibition received by other simple cells')
        #ax.plot(arge, avg_thness_spikes, 'k-', label="spikes without inhibition")
        #p_.plot(arge, avg_thness_spikes2, 'r-', label="spikes with inhibition")

        print((avg_thness_lat2).shape)
        print((avg_thness_std_td_err2).shape)
        print((avg_thness_std_lat_err2).shape)

        plt.axvline(x = np.argmax(avg_thness_spikes2), color = 'k', linestyle = 'dashed', label="moment at which the max spike was reached")    

        """for val in range(len(arge)):
            if(val==0):
                #p_.plot( [arge[val], arge[val]], [avg_thness_lat[val]-avg_thness_std_lat_err[val],avg_thness_lat[val]+avg_thness_std_lat_err[val]], 'r--',label='standard error of lateral inhibition')
                #p_.plot([arge[val], arge[val]], [avg_thness_td[val]-avg_thness_std_td_err[val],avg_thness_td[val]+avg_thness_std_td_err[val]], 'g--', label='standard error of top-down inhibition')
                #ax.plot([arge[val], arge[val]], [avg_thness_spikes[val]-avg_thness_std_spikes_err[val],avg_thness_spikes[val]+avg_thness_std_spikes_err[val]], 'k--', label='standard error of spikes (without inhibition)')
                
                #ax.plot( [arge[val], arge[val]], [avg_thness_lat2[val]-avg_thness_std_lat_err2[val],avg_thness_lat2[val]+avg_thness_std_lat_err2[val]], 'r--',label='standard error of lateral inhibition')
                #ax.plot([arge[val], arge[val]], [avg_thness_td2[val]-avg_thness_std_td_err2[val],avg_thness_td2[val]+avg_thness_std_td_err2[val]], 'g--', label='standard error of top-down inhibition')
                #p_.plot([arge[val], arge[val]], [avg_thness_spikes2[val]-avg_thness_std_spikes_err2[val],avg_thness_spikes2[val]+avg_thness_std_spikes_err2[val]], 'r--', label='standard error of spikes (with inhibition)')
                print("hey")
            else:
                #p_.plot( [arge[val], arge[val]], [avg_thness_lat[val]-avg_thness_std_lat_err[val],avg_thness_lat[val]+avg_thness_std_lat_err[val]], 'r--')
                #p_.plot([arge[val], arge[val]], [avg_thness_td[val]-avg_thness_std_td_err[val],avg_thness_td[val]+avg_thness_std_td_err[val]], 'g--')
                #ax.plot([arge[val], arge[val]], [avg_thness_spikes[val]-avg_thness_std_spikes_err[val],avg_thness_spikes[val]+avg_thness_std_spikes_err[val]], 'k--')
        
                ax.plot( [arge[val], arge[val]], [avg_thness_lat2[val]-avg_thness_std_lat_err2[val],avg_thness_lat2[val]+avg_thness_std_lat_err2[val]], 'r--')
                ax.plot([arge[val], arge[val]], [avg_thness_td2[val]-avg_thness_std_td_err2[val],avg_thness_td2[val]+avg_thness_std_td_err2[val]], 'g--')
                #p_.plot([arge[val], arge[val]], [avg_thness_spikes2[val]-avg_thness_std_spikes_err2[val],avg_thness_spikes2[val]+avg_thness_std_spikes_err2[val]], 'r--')
        #suppr_percentage = suppression_metric(avg_thness_spikes2)
        #ax.text(31, 1, "Suppression of " + str(int(suppr_percentage)) + "%", fontsize = 42)
        """
        plt.xticks(arge)
        fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
        return avg_thness_std_lat_err2