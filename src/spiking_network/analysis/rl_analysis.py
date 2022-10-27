#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 23 2022

@author: thomas
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from natsort import natsorted

from src.spiking_network.analysis.spike_train import fast_time_histogram
from src.spiking_network.network.neuvisys import SpikingNetwork


def gaussian(a, b, c, x):
    return a * np.exp(-0.5 * (x - b) ** 2 / c ** 2)


def value_plot(spinet, display_score=False):
    subset = slice(0, 200000)
    reward = np.array(spinet.state["learning_data"]["reward"][subset])
    value = np.array(spinet.state["learning_data"]["value"][subset])
    value_dot = np.array(spinet.state["learning_data"]["valueDot"][subset])
    td_error = np.array(spinet.state["learning_data"]["tdError"][subset])
    if display_score:
        score = np.array(spinet.state["learning_data"]["score"][subset])
    actions = []
    for i in range(len(spinet.rl_conf["actionMapping"])):
        actions.append(np.array(spinet.state["learning_data"]["action_" + str(i)][subset]))
    t = np.arange(0, reward.size) * 1e-3

    plt.figure(figsize=(40, 8))
    plt.title("Reward and value curves")
    plt.xlabel("time (s)")
    plt.ylabel("Reward / Value")
    plt.plot(t, reward, label="reward")
    plt.plot(t, value, label="value")
    # plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("Action decisions")
    plt.xlabel("time (s)")
    plt.ylabel("Number of spikes")
    actions_label = ["left", "stop", "right"]
    for i, action in enumerate(actions):
        # plt.plot(t, action, label="action " + str(i))
        plt.plot(t, action, label=actions_label[i])
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("Value derivative")
    plt.xlabel("time (s)")
    plt.ylabel("Value derivative")
    plt.plot(t, value_dot, color="green", label="value_dot")
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("TD error")
    plt.xlabel("time (s)")
    plt.ylabel("TD error")
    plt.plot(t, td_error, color="red", label="td_error")
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.show()

    if display_score:
        plt.figure(figsize=(40, 8))
        plt.title("Score")
        plt.xlabel("time (s)")
        plt.plot(10 * np.arange(0, score.shape[0]), score, color="red", label="td_error")
        plt.show()


def policy_plot(spinet, action_bin, action_labels):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    angle_color = colors[-3]

    rewards = np.array(spinet.state["learning_data"]["reward"])
    actions = np.array(spinet.state["learning_data"]["action"], dtype=np.int32)
    actions_colors = np.array(colors)[actions]
    nb_actions = np.unique(actions).size

    sp_trains = []
    for i in range(nb_actions):
        sp_train = spinet.spikes[-1][i * 50:(i + 1) * 50].flatten()
        sp_train = sp_train[sp_train != 0]
        sp_train = np.sort(sp_train)
        sp_trains.append(sp_train)

    hist_bin = np.arange(0, sp_trains[0][-1], int(1e3 * action_bin))
    activity_variations = []
    for i in range(nb_actions):
        activity_variations.append(np.histogram(sp_trains[i], bins=hist_bin)[0])
    x_size = activity_variations[0].size

    m = rewards.size // x_size + 1
    rewards = rewards[::m]
    t = np.linspace(0, activity_variations[0].size, actions_colors.size)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 20))
    ax1bis = ax1.twinx()
    for i in range(nb_actions):
        ax1.plot(activity_variations[i], color=colors[i], label=action_labels[i])
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Number of spikes")
    # ax1.vlines(t, 0, np.max(activity_variations), color=actions_colors, alpha=0.5)
    ax1bis.plot(rewards, color=angle_color, label="rotation angle")
    ax1bis.tick_params(axis="y", labelcolor=angle_color)
    ax1bis.set_ylabel("Rotation angle (°)", color=angle_color)
    ax1.legend(loc="upper right")

    ax1.axvspan(0, rewards.size // 2 - 1, color="orange", alpha=0.2)
    ax1.axvspan(rewards.size // 2 - 1, rewards.size, color="blue", alpha=0.2)

    action_values = []
    for i in range(nb_actions):
        action_values.append(np.array(spinet.state["learning_data"]["action_" + str(i)]))

    # angles = angles[:action_values[0].size]
    # t = np.linspace(0, action_values[0].size, actions_colors.size)
    #
    # ax2bis = ax2.twinx()
    # for i in range(nb_actions):
    #     ax2.plot(action_values[i], color=colors[i], label=action_labels[i])
    # ax2.set_xlabel("Time (ms)")
    # ax2.set_ylabel("Number of spikes")
    # ax2.vlines(t, 0, np.max(action_values), color=actions_colors, alpha=0.5)
    # ax2bis.plot(angles, color=angle_color, label="rotation angle")
    # ax2bis.set_ylim(0)
    # ax2bis.tick_params(axis="y", labelcolor=angle_color)
    # ax2bis.set_ylabel("Rotation angle (°)", color=angle_color)
    #
    # boole = angles > 90
    # ax2.axvspan(0, np.argmax(angles > 90), color="orange", alpha=0.2)
    # ax2.axvspan(np.argmax(angles > 90), np.where(boole)[0][-1], color="blue", alpha=0.2)
    # ax2.axvspan(np.where(boole)[0][-1], angles.size, color="orange", alpha=0.2)
    # ax2.legend(loc="upper right")


def score_tracking(spinet):
    actions_selections = np.array(spinet.state["learning_data"]["action"], dtype=np.int32)
    size = actions_selections.size
    perfect = np.zeros(size)
    perfect[size // 2:] = 1

    return np.sum(np.abs(perfect - actions_selections))


def score_stabilisation(spinet):
    actions_selections = np.array(spinet.state["learning_data"]["action"], dtype=np.int32)
    size = actions_selections.size
    perfect = np.ones(size)
    perfect[:size // 4] = 0
    perfect[3 * (size // 4):] = 0

    return np.sum(np.abs(perfect - actions_selections))


def policy_plot_multiple(spinet, action_bin, nb_actions):
    sp_trains = []
    for i in range(nb_actions):
        sp_train = spinet.spikes[-1][i * 50:(i + 1) * 50].flatten()
        sp_train = sp_train[sp_train != 0]
        sp_train = np.sort(sp_train)
        sp_trains.append(sp_train)

    hist_bin = np.arange(0, sp_trains[0][-1], int(1e3 * action_bin))
    activity_variations = []
    for i in range(nb_actions):
        activity_variations.append(np.histogram(sp_trains[i], bins=hist_bin)[0])
    return activity_variations


def value_plot_multiple(spinet, value_bin):
    sp_train = spinet.spikes[-2].flatten()
    sp_train = sp_train[sp_train != 0]
    sp_train = np.sort(sp_train)
    hist_bin = np.arange(0, sp_train[-1], int(1e3 * value_bin))
    return np.histogram(sp_train, bins=hist_bin)[0]


def validation_plot(values, policies, scores, action_labels, task):
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    fig2, ax2 = plt.subplots(1, 1, figsize=(40, 20))

    nb_curves = len(values)
    value_colors = pl.cm.jet(np.linspace(0, 1, nb_curves))
    action_colors = [pl.cm.Blues(np.linspace(0.2, 1, nb_curves)), pl.cm.Reds(np.linspace(0.2, 1, nb_curves))]

    for i in range(nb_curves):
        t = np.arange(values[i].size)
        if i == nb_curves - 1:
            ax2.plot(t, values[i], color=value_colors[i], label="Value")
        else:
            ax2.plot(t, values[i], color=value_colors[i])
        ax2.set_ylabel("Number of spikes")

        t = np.arange(policies[i][0].size)
        if i == nb_curves - 1:
            for j in range(2):
                ax.plot(t, policies[i][j], color=action_colors[j][i], label=action_labels[j])
        else:
            for j in range(2):
                ax.plot(t, policies[i][j], color=action_colors[j][i])
            ax.set_ylabel("Number of spikes")

    print(t.size)

    if task == "tracking":
        ax.axvspan(0, 4, color=action_colors[0][-1], alpha=0.1)
        ax.axvspan(4, 8, color=action_colors[1][-1], alpha=0.1)
    elif task == "orientation":
        ax.axvspan(0, 5, color=action_colors[0][-1], alpha=0.1)
        ax.axvspan(5, 10.5, color=action_colors[1][-1], alpha=0.1)
        ax.axvspan(10.5, 16, color=action_colors[0][-1], alpha=0.1)
        ax.axvspan(16, 21, color=action_colors[1][-1], alpha=0.1)

    if task == "tracking":
        ax.set_xticks(t, np.round(np.linspace(-173, 173, t.size), 1))
        ax2.set_xticks(t, np.round(np.linspace(-173, 173, t.size), 1))
        ax.set_xlabel("Distance to visual center (px)")
        ax2.set_xlabel("Distance to visual center (px)")
    elif task == "orientation":
        ax.set_xticks(t, np.round(np.linspace(0, 360, t.size), 1))
        ax2.set_xticks(t, np.round(np.linspace(0, 360, t.size), 1))
        ax2.set_xlabel("Orientation (°)")
        ax2.set_xlabel("Orientation (°)")

    ax.legend(loc='best')
    ax2.legend(loc='best')
    plt.show()

    plt.figure(figsize=(40, 20))
    plt.title("Evolution of the policy error during training")
    plt.xlabel("Trials")
    plt.ylabel("Error from perfect policy")
    plt.plot(scores, color="#6D326D", label="Policy error")
    plt.legend()


def value_policy_evaluation(folder, binsize, nb_action):
    scores = []
    values, policies = [], []
    for i, network_path in enumerate(natsorted(os.listdir(folder))):
        spinet = SpikingNetwork(folder + network_path + "/", loading=[False, False, True, True])

        values.append(value_plot_multiple(spinet, binsize))
        policies.append(policy_plot_multiple(spinet, binsize, nb_action))
        scores.append(score_tracking(spinet))
    return values, policies, scores


def test_validation(folder):
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    fig2, ax2 = plt.subplots(1, 1, figsize=(40, 20))

    nb_curves = len(natsorted(os.listdir(folder)))
    colors = pl.cm.jet(np.linspace(0, 1, nb_curves))

    action1_colors = pl.cm.Blues(np.linspace(0.2, 1, nb_curves))
    action2_colors = pl.cm.Reds(np.linspace(0.2, 1, nb_curves))

    x = np.linspace(0, 346, 9)

    for i, network_path in enumerate(natsorted(os.listdir(folder))):
        spinet = SpikingNetwork(folder + network_path + "/", loading=[False, False, True, True])

        critic_sum = critic_validation(spinet)
        value_1_sum, value_2_sum = value_validation(spinet)

        ax.plot(x, critic_sum, color=colors[i])
        ax2.plot(x, value_1_sum, color=action1_colors[i])
        ax2.plot(x, value_2_sum, color=action2_colors[i])


def critic_validation(spinet):
    spikes = spinet.spikes[2]
    spikes = spikes[spikes > 0]
    spikes = np.sort(spikes)

    intervals = np.arange(0, 4500001, 500000)
    sum = []
    for i in range(intervals.size - 1):
        sum.append(np.sum(spikes[(spikes >= intervals[i]) & (spikes <= intervals[i + 1])]))
    return sum


def value_validation(spinet):
    spikes_0 = spinet.spikes[3][0:50]
    spikes_0 = spikes_0[spikes_0 > 0]
    spikes_0 = np.sort(spikes_0)

    spikes_1 = spinet.spikes[3][50:100]
    spikes_1 = spikes_1[spikes_1 > 0]
    spikes_1 = np.sort(spikes_1)

    intervals = np.arange(0, 4500001, 500000)
    sum_0 = []
    sum_1 = []
    for i in range(intervals.size - 1):
        sum_0.append(np.sum(spikes_0[(spikes_0 >= intervals[i]) & (spikes_0 <= intervals[i + 1])]))
        sum_1.append(np.sum(spikes_1[(spikes_1 >= intervals[i]) & (spikes_1 <= intervals[i + 1])]))
    return sum_0, sum_1


def full_validation(folder):
    values = []
    actions = []
    for net in os.listdir(folder):
        network_path = folder + net + "/"
        spinet = SpikingNetwork(network_path, loading=[True, True, True, True])
        reward, hist_value, hist_action = validation_critic_actor2(spinet, display=False)
        values.append(hist_value)
        actions.append(hist_action)

    values = np.array(values)
    actions = np.array(actions)
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    mean_actions = np.mean(actions, axis=0)
    # mean_actions[1] += gaussian(35, 0, 0.4, 4 * reward[:-1])
    std_actions = np.std(actions, axis=0)
    low_values = mean_values - std_values
    high_values = mean_values + std_values
    low_actions = []
    high_actions = []
    for i in range(mean_actions.shape[0]):
        low_actions.append(mean_actions[i] - std_actions[i])
        high_actions.append(mean_actions[i] + std_actions[i])

    plt.figure()
    plt.xlabel("Ball angular error (degree)")
    plt.ylabel("Value / Reward")
    plt.vlines(0, 0, 80, colors="red", linestyles="dashed", alpha=0.5)
    plt.plot(reward[:-1], mean_values, label="value", color="#5DA9E9")
    plt.plot(reward[:-1], gaussian(80, 0, 0.4, 4 * reward[:-1]), label="reward", color="#6D326D")
    plt.fill_between(reward[:-1], low_values, high_values, alpha=0.5)
    plt.legend()

    plt.figure()
    plt.xlabel("Ball angular error (degree)")
    plt.ylabel("Number of spikes")
    plt.vlines(0, 0, 600, colors="red", linestyles="dashed", alpha=0.5)
    plt.plot(reward[:-1], mean_actions[0], label="left")
    plt.plot(reward[:-1], mean_actions[1], label="right")
    # plt.plot(reward[:-1], mean_actions[2], label="right")
    plt.fill_between(reward[:-1], low_actions[0], high_actions[0], alpha=0.5)
    plt.fill_between(reward[:-1], low_actions[1], high_actions[1], alpha=0.5)
    # plt.fill_between(reward[:-1], low_actions[2], high_actions[2], alpha=0.5)
    plt.legend()


def validation_critic_actor(spinet, display=True):
    reward = np.array(spinet.state["learning_data"]["error"])
    time = np.array(spinet.state["learning_data"]["time"])
    bins = 100

    middle = np.argwhere(np.diff(reward) < 0)[0][0]
    reward1 = reward[:middle]
    reward2 = reward[middle:]
    time1 = time[:middle]
    time2 = time[middle:]

    idx1 = np.round(np.linspace(0, len(time1) - 1, bins)).astype(int)
    time1 = time1[idx1]
    # time1 = np.concatenate(([0], time1))
    reward1 = reward1[idx1]

    hist_left1 = fast_time_histogram(spinet.spikes[3][0:50], time1)
    hist_stop1 = fast_time_histogram(spinet.spikes[3][50:100], time1)
    hist_right1 = fast_time_histogram(spinet.spikes[3][100:150], time1)

    if display:
        plt.figure()
        plt.plot(reward1[1:], hist_left1)
        plt.plot(reward1[1:], hist_stop1)
        plt.plot(reward1[1:], hist_right1)

    idx2 = np.round(np.linspace(0, len(time2) - 1, bins)).astype(int)
    time2 = time2[idx2]
    reward2 = reward2[idx2]

    hist_left2 = fast_time_histogram(spinet.spikes[3][0:50], time2)
    hist_stop2 = fast_time_histogram(spinet.spikes[3][50:100], time2)
    hist_right2 = fast_time_histogram(spinet.spikes[3][100:150], time2)

    if display:
        plt.figure()
        plt.plot(reward2[1:], hist_left2)
        plt.plot(reward2[1:], hist_stop2)
        plt.plot(reward2[1:], hist_right2)

    hist_left = (hist_left1 + hist_left2[::-1]) / 2
    hist_stop = (hist_stop1 + hist_stop2[::-1]) / 2
    hist_right = (hist_right1 + hist_right2[::-1]) / 2

    hist_value1 = fast_time_histogram(spinet.spikes[2], time1)
    hist_value2 = fast_time_histogram(spinet.spikes[2], time2)
    hist_value = (hist_value1 + hist_value2[::-1]) / 2
    hist_value = 80 * hist_value / np.max(hist_value)

    if display:
        plt.figure()
        plt.xlabel("Ball angular error (degree)")
        plt.ylabel("Value / Reward")
        plt.vlines(0, 0, np.max(hist_value), colors="red", linestyles="dashed", alpha=0.5)
        plt.plot(reward1[:-1], hist_value, label="value", color="#5DA9E9")
        plt.plot(reward1[:-1], gaussian(80, 0, 0.4, 4 * reward1[:-1]), label="reward", color="#6D326D")
        plt.legend()

        plt.figure()
        plt.xlabel("Ball angular error (degree)")
        plt.ylabel("Number of spikes")
        plt.vlines(0, 0, np.max(hist_right), colors="red", linestyles="dashed", alpha=0.5)
        plt.plot(reward1[:-1], hist_left, label="left")
        plt.plot(reward1[:-1], hist_stop, label="stop")
        plt.plot(reward1[:-1], hist_right, label="right")
        plt.legend()

    return reward1, hist_value, [hist_left, hist_stop, hist_right]


def validation_critic_actor2(spinet, display=True):
    reward = np.array(spinet.state["learning_data"]["error"])
    time = np.array(spinet.state["learning_data"]["time"])
    bins = 100

    middle = np.argwhere(np.diff(reward) < 0)[0][0]
    reward1 = reward[:middle]
    reward2 = reward[middle:]
    time1 = time[:middle]
    time2 = time[middle:]

    idx1 = np.round(np.linspace(0, len(time1) - 1, bins)).astype(int)
    time1 = time1[idx1]
    # time1 = np.concatenate(([0], time1))
    reward1 = reward1[idx1]

    hist_left1 = fast_time_histogram(spinet.spikes[3][0:50], time1)
    hist_right1 = fast_time_histogram(spinet.spikes[3][50:100], time1)

    if display:
        plt.figure()
        plt.plot(reward1[1:], hist_left1)
        plt.plot(reward1[1:], hist_right1)

    idx2 = np.round(np.linspace(0, len(time2) - 1, bins)).astype(int)
    time2 = time2[idx2]
    reward2 = reward2[idx2]

    hist_left2 = fast_time_histogram(spinet.spikes[3][0:50], time2)
    hist_right2 = fast_time_histogram(spinet.spikes[3][50:100], time2)

    if display:
        plt.figure()
        plt.plot(reward2[1:], hist_left2)
        plt.plot(reward2[1:], hist_right2)

    hist_left = (hist_left1 + hist_left2[::-1]) / 2
    hist_right = (hist_right1 + hist_right2[::-1]) / 2

    hist_value1 = fast_time_histogram(spinet.spikes[2], time1)
    hist_value2 = fast_time_histogram(spinet.spikes[2], time2)
    hist_value = (hist_value1 + hist_value2[::-1]) / 2
    hist_value = 80 * hist_value / np.max(hist_value)

    if display:
        plt.figure()
        plt.xlabel("Ball angular error (degree)")
        plt.ylabel("Value / Reward")
        plt.vlines(0, 0, np.max(hist_value), colors="red", linestyles="dashed", alpha=0.5)
        plt.plot(reward1[:-1], hist_value, label="value", color="#5DA9E9")
        plt.plot(reward1[:-1], gaussian(80, 0, 0.4, 4 * reward1[:-1]), label="reward", color="#6D326D")
        plt.legend()

        plt.figure()
        plt.xlabel("Ball angular error (degree)")
        plt.ylabel("Number of spikes")
        plt.vlines(0, 0, np.max(hist_right), colors="red", linestyles="dashed", alpha=0.5)
        plt.plot(reward1[:-1], hist_left, label="left")
        plt.plot(reward1[:-1], hist_right, label="right")
        plt.legend()

    return reward1, hist_value, [hist_left, hist_right]
