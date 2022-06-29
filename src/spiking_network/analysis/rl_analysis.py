#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 23 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt


def value_plot(spinet):
    reward = np.array(spinet.state["learning_data"]["reward"])
    value = np.array(spinet.state["learning_data"]["value"])
    value_dot = np.array(spinet.state["learning_data"]["valueDot"])
    td_error = np.array(spinet.state["learning_data"]["tdError"])
    # score = np.array(spinet.state["learning_data"]["score"])
    # t = np.linspace(0, np.max(spinet.spikes[0]), reward.size) * 1e-6
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

    # plt.figure(figsize=(40, 8))
    # plt.title("Score")
    # plt.xlabel("time (s)")
    # plt.plot(10 * np.arange(0, score.shape[0]), score, color="red", label="td_error")
    # plt.show()


def policy_plot(spinet):
    actions = np.array(spinet.state["learning_data"]["action"], dtype=np.int32)
    rewards = np.array(spinet.state["learning_data"]["reward"])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    angle_color = colors[-3]
    actions_colors = np.array(colors)[actions]

    nb_actions = np.unique(actions).size

    action_values = []
    for i in range(nb_actions):
        action_values.append(np.array(spinet.state["learning_data"]["action_" + str(i)]))
    exploration = np.array(spinet.state["learning_data"]["exploration"])

    bins = 50
    sp_trains = []
    for i in range(nb_actions):
        sp_train = spinet.spikes[-1][i * 50:(i + 1) * 50].flatten()
        sp_train = sp_train[sp_train != 0]
        sp_train = np.sort(sp_train)
        sp_trains.append(sp_train)

    hist_bin = np.arange(0, sp_trains[0][-1], int(1e3 * bins))
    activity_variations = []
    for i in range(nb_actions):
        activity_variations.append(np.histogram(sp_trains[i], bins=hist_bin)[0])

    angles = np.concatenate(
        [np.linspace(0, 180, activity_variations[0].size // 2 + 1),
         np.linspace(180, 0, activity_variations[0].size // 2)[1:]])
    t = np.arange(0, activity_variations[0].size)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1bis = ax1.twinx()
    for i in range(nb_actions):
        ax1.plot(activity_variations[i], color=colors[i], label="action " + str(i))
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Number of spikes")
    ax1.vlines(t, 0, np.max(activity_variations), color=actions_colors, linestyles="dotted")
    ax1bis.plot(angles, color=angle_color, label="rotation angle")
    ax1bis.tick_params(axis="y", labelcolor=angle_color)
    ax1bis.set_ylabel("Rotation angle (°)", color=angle_color)
    ax1.legend(loc="upper right")

    angles = np.concatenate(
        [np.linspace(0, 180, action_values[0].size // 2), np.linspace(180, 0, action_values[0].size // 2)])
    t = np.linspace(50, action_values[0].size-15, activity_variations[0].size)

    ax2bis = ax2.twinx()
    for i in range(nb_actions):
        ax2.plot(action_values[i], color=colors[i], label="action " + str(i))
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Number of spikes")
    ax2.vlines(t, 0, np.max(action_values), color=actions_colors, linestyles="dotted")
    ax2bis.plot(angles, color=angle_color, label="rotation angle")
    ax2bis.tick_params(axis="y", labelcolor=angle_color)
    ax2bis.set_ylabel("Rotation angle (°)", color=angle_color)

    angles = np.concatenate(
        [np.linspace(0, 180, activity_variations[0].size // 2 + 1),
         np.linspace(180, 0, activity_variations[0].size // 2)[1:]])
    t = np.arange(0, activity_variations[0].size)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
    theta = np.linspace(0, 2 * np.pi, activity_variations[0].size)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    for i in range(nb_actions):
        ax1.plot(theta, activity_variations[i], color=colors[i], label="action " + str(i))
    ax1.set_xlabel("Time (ms)")
    ax1.vlines(t, 0, np.max(activity_variations), color=actions_colors, linestyles="dotted")

    angles = np.concatenate(
        [np.linspace(0, 180, action_values[0].size // 2), np.linspace(180, 0, action_values[0].size // 2)])
    t = np.arange(50, action_values[0].size, 50)

    theta = np.linspace(0, 2 * np.pi, action_values[0].size)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    for i in range(nb_actions):
        ax2.plot(theta, action_values[i], color=colors[i], label="action " + str(i))
    ax2.set_xlabel("Time (ms)")
    ax2.vlines(t, 0, np.max(action_values), color=actions_colors[::-1], linestyles="dotted")
    plt.show()
