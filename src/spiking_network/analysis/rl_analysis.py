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
from src.spiking_network.network.neuvisys import SpikingNetwork


def value_plot(spinet, display_score=False):
    reward = np.array(spinet.state["learning_data"]["reward"])
    value = np.array(spinet.state["learning_data"]["value"])
    value_dot = np.array(spinet.state["learning_data"]["valueDot"])
    td_error = np.array(spinet.state["learning_data"]["tdError"])
    if display_score:
        score = np.array(spinet.state["learning_data"]["score"])
    actions = []
    for i in range(len(spinet.rl_conf["actionMapping"])):
        actions.append(np.array(spinet.state["learning_data"]["action_" + str(i)]))
    t = np.arange(0, reward.size) * 1e-3

    plt.figure(figsize=(40, 8))
    plt.title("Reward and value curves")
    plt.xlabel("time (s)")
    plt.plot(t, reward, label="reward")
    plt.plot(t, value, label="value")
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("Action decisions")
    plt.xlabel("time (s)")
    for i, action in enumerate(actions):
        plt.plot(t, action, label="action " + str(i))
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("Value derivative")
    plt.xlabel("time (s)")
    plt.plot(t, value_dot, color="green", label="value_dot")
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.show()

    plt.figure(figsize=(40, 8))
    plt.title("TD error")
    plt.xlabel("time (s)")
    plt.plot(t, td_error, color="red", label="td_error")
    plt.hlines(0, 0, t[-1], linestyles="dashed")
    plt.show()

    td_actions = []
    for i in range(0, td_error.shape[0], 10):
        td_actions.append(np.mean(td_error[i:i + 10]))

    plt.figure(figsize=(40, 8))
    plt.title("mean TD error at action choice")
    plt.xlabel("time (s)")
    plt.plot(td_actions, color="cyan", label="td_error")
    plt.hlines(0, 0, i / 10, linestyles="dashed")
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
        if i == nb_curves-1:
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
    for i in range(intervals.size-1):
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
    for i in range(intervals.size-1):
        sum_0.append(np.sum(spikes_0[(spikes_0 >= intervals[i]) & (spikes_0 <= intervals[i + 1])]))
        sum_1.append(np.sum(spikes_1[(spikes_1 >= intervals[i]) & (spikes_1 <= intervals[i + 1])]))
    return sum_0, sum_1