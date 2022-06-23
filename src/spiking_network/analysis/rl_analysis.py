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
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    actions_colors = np.array(colors)[actions]

    nb_actions = np.unique(actions).size

    action_values = []
    for i in range(nb_actions):
        action_values.append(np.array(spinet.state["learning_data"]["action_" + str(i)]))
    exploration = np.array(spinet.state["learning_data"]["exploration"])
    angles = np.concatenate([np.linspace(0, 180, action_values[0].size // 2), np.linspace(180, 0, action_values[0].size // 2)])

    angle_color = "#69b3a2"
    t = np.arange(50, action_values[0].size, 50)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(nb_actions):
        ax1.plot(action_values[i], color=colors[i], label="action " + str(i))
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Actors number of spikes")
    ax1.vlines(t[:-1], 0, np.max(action_values), color=actions_colors, linestyles="dotted")
    ax2.plot(angles, color=angle_color, label="rotation angle")
    ax2.tick_params(axis="y", labelcolor=angle_color)
    ax2.set_ylabel("Rotation angle (degree)", color=angle_color)
    plt.show()

    theta = np.linspace(0, 2*np.pi, action_values[0].size)
    fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    # ax2 = ax1.twinx()
    for i in range(nb_actions):
        ax1.plot(theta, action_values[i], color=colors[i], label="action " + str(i))
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Actors number of spikes")
    ax1.vlines(t[:-1], 0, np.max(action_values), color=actions_colors, linestyles="dotted")
    # ax2.plot(angles, color=angle_color, label="rotation angle")
    ax2.tick_params(axis="y", labelcolor=angle_color)
    ax2.set_ylabel("Rotation angle (degree)", color=angle_color)
    plt.show()