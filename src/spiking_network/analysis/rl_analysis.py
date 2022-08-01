#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 23 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt


def value_plot(spinet, display_score=False):
    reward = np.array(spinet.state["learning_data"]["reward"])
    value = np.array(spinet.state["learning_data"]["value"])
    value_dot = np.array(spinet.state["learning_data"]["valueDot"])
    td_error = np.array(spinet.state["learning_data"]["tdError"])
    if display_score:
        score = np.array(spinet.state["learning_data"]["score"])
    actions = []
    for i in range(len(spinet.rl_conf["actionMapping"])):
        actions.append(np.array(spinet.state["learning_data"]["action_"+str(i)]))
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
    plt.title("Reward and value curves")
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
    # angles = np.rad2deg(np.array(spinet.state["learning_data"]["angle"]))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    angle_color = colors[-3]

    actions = np.array(spinet.state["learning_data"]["action"], dtype=np.int32)
    actions_colors = np.array(colors)[actions]

    nb_actions = np.unique(actions).size

    action_values = []
    for i in range(nb_actions):
        action_values.append(np.array(spinet.state["learning_data"]["action_" + str(i)]))

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

    t = np.linspace(0, activity_variations[0].size, actions_colors.size)

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(40, 20))
    ax1bis = ax1.twinx()
    for i in range(nb_actions):
        ax1.plot(activity_variations[i], color=colors[i], label=action_labels[i])
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Number of spikes")
    ax1.vlines(t, 0, np.max(activity_variations), color=actions_colors, alpha=0.5)
    # ax1bis.plot(angles, color=angle_color, label="rotation angle")
    ax1bis.tick_params(axis="y", labelcolor=angle_color)
    ax1bis.set_ylabel("Rotation angle (°)", color=angle_color)
    ax1.legend(loc="upper right")

    # angles = angles[:action_values[0].size]
    t = np.linspace(0, action_values[0].size, actions_colors.size)

    ax2bis = ax2.twinx()
    for i in range(nb_actions):
        ax2.plot(action_values[i], color=colors[i], label=action_labels[i])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Number of spikes")
    ax2.vlines(t, 0, np.max(action_values), color=actions_colors, alpha=0.5)
    # ax2bis.plot(angles, color=angle_color, label="rotation angle")
    ax2bis.set_ylim(0)
    ax2bis.tick_params(axis="y", labelcolor=angle_color)
    ax2bis.set_ylabel("Rotation angle (°)", color=angle_color)

    # boole = angles > 90
    # ax2.axvspan(0, np.argmax(angles > 90), color="orange", alpha=0.2)
    # ax2.axvspan(np.argmax(angles > 90), np.where(boole)[0][-1], color="blue", alpha=0.2)
    # ax2.axvspan(np.where(boole)[0][-1], angles.size, color="orange", alpha=0.2)
    ax2.legend(loc="upper right")
