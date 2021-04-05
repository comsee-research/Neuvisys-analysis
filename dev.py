#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:47:24 2020

@author: alphat
"""

import os
from fpdf import FPDF
from natsort import natsorted
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import panda as pd
import cv2 as cv
from spiking_network.spiking_network import SpikingNetwork

#%%

path = "/media/alphat/SSD Games/Thesis/configuration/network_"

a = []
for i in range(8):
    with open(path + str(i) + "/configs/pooling_neuron_config.json") as file:
        a.append(json.load(file))

df = pd.DataFrame(a)


#%%

network_path = ""
spinet = SpikingNetwork(network_path)


#%% Spike rate

network_path = "/home/alphat/neuvisys-dv/configuration/network/"
time = np.max(spinet.sspikes)

srates = np.count_nonzero(spinet.sspikes, axis=1) / (time * 1e-6)
print("mean:", np.mean(srates))
print("std:", np.std(srates))


#%%

row = 6
col = 6
im_size = (20, 20)
im_size_pad = (22, 22)

pdf = FPDF("P", "mm", (col * im_size_pad[0], row * im_size_pad[1]))
pdf.add_page()

path = "/home/alphat/neuvisys-dv/configuration/NETWORKS/complex_selectivity/figures/complex_directions/"

images = natsorted(os.listdir(path))  # [::-1]
random.shuffle(images)

count = 0
for i in range(row):
    for j in range(col):
        pdf.image(
            path + images[count],
            x=j * im_size_pad[0],
            y=i * im_size_pad[1],
            w=im_size[0],
            h=im_size[1],
        )
        count += 1

pdf.output("/home/alphat/Desktop/images/directions.pdf", "F")


#%%

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("none")
ax.grid(color="grey", axis="y", linestyle="-", linewidth=0.25, alpha=0.5)

color1 = dict(color="#2C363F")
color2 = dict(color="#9E7B9B")

ax.set_ylabel("normalized vector length")
ax.boxplot(
    np.abs(des),
    positions=[0],
    labels=["direction exponential"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(oes),
    positions=[1],
    labels=["orientation exponential"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(dls),
    positions=[2],
    labels=["direction linear"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(ols),
    positions=[3],
    labels=["orientation linear"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(dsr),
    positions=[4],
    labels=["direction step left"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(osr),
    positions=[5],
    labels=["orientation step left"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(dsl),
    positions=[6],
    labels=["direction step right"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(osl),
    positions=[7],
    labels=["orientation step right"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(dss),
    positions=[8],
    labels=["direction step sym"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(oss),
    positions=[9],
    labels=["orientation step sym"],
    boxprops=color1,
    medianprops=color2,
    whiskerprops=color1,
    capprops=color1,
    flierprops=dict(markeredgecolor=color1["color"]),
)
plt.show()


#%%

fig, ax = plt.subplots(figsize=(8, 12))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

color1 = dict(color="#2C363F")
color2 = dict(color="#9E7B9B")
ax.boxplot(np.abs(dir_vec), positions=[0], labels=["direction space"], boxprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), medianprops=dict(linestyle='-', linewidth=1.7, color="#9E7B9B"), whiskerprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), capprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(ori_vec), positions=[1], labels=["orientation space"], boxprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), medianprops=dict(linestyle='-', linewidth=1.7, color="#9E7B9B"), whiskerprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), capprops=dict(linestyle='-', linewidth=1.7, color="#2C363F"), flierprops=dict(markeredgecolor=color1["color"]))
ax.set_ylabel("Normalized vector length", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
plt.savefig("/home/alphat/Desktop/images/boxplot_ori_dir.png", bbox_inches="tight")

#%%

plt.figure(figsize=(8, 12))
ax = sns.histplot((np.angle(ori_vec))*180/np.pi, bins=np.linspace(-180, 180, 16), color="#2C363F")
ax.set_xlabel("Orientation (degree)", fontsize=18)
ax.set_ylabel("Count", fontsize=18)
ax.set_xticks(np.arange(-180, 181, 45))
ax.set_xtickslabels(np.arange(-90, 91, 45))
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

plt.savefig("/home/alphat/Desktop/images/hist_ori.png", bbox_inches="tight")

#%%

eta_ltp = 0.2
eta_ltd = -0.2
tau_ltp = 20
tau_ltd = 20

t = np.linspace(-80, 80, 10000)

step = np.zeros(10000)
step[(t > -tau_ltd) & (t <= 0)] = -eta_ltd
step[(t >= 0) & (t < tau_ltd)] = eta_ltp

lin = np.zeros(10000)
lin[(t > -tau_ltd) & (t <= 0)] = (
    -eta_ltd
    * -t[(t > -tau_ltd) & (t <= 0)][::-1]
    / np.max(-t[(t > -tau_ltd) & (t <= 0)][::-1])
)
lin[(t >= 0) & (t < tau_ltd)] = (
    eta_ltp
    * t[(t >= 0) & (t < tau_ltd)][::-1]
    / np.max(t[(t >= 0) & (t < tau_ltd)][::-1])
)

exp = np.concatenate(
    (eta_ltd * np.exp(t[t < 0] / tau_ltd), eta_ltp * np.exp(-t[t >= 0] / tau_ltp))
)


fig, axs = plt.subplots(1, 3)

for ax in axs.flat:
    ax.grid(alpha=0.2, linestyle="--")
    ax.axhline(0, alpha=0.3, linestyle="-", color="k")
    ax.axvline(0, alpha=0.3, linestyle="-", color="k")
    ax.set_ylim(-0.3, 0.3)
    ax.set_ylabel("Synaptic change (mV)")
    ax.label_outer()

axs[1].set_xlabel("Spike timing (ms)")

axs[0].plot(t, step, color="k")
axs[0].set_title("Step")
axs[1].plot(t, lin, color="k")
axs[1].set_title("Linear")
axs[2].plot(t, exp, color="k")
axs[2].set_title("Exponential")


#%%

block_size = 3
min_disp = 0
num_disp = 16

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=1,
    preFilterCap=0,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=1,
    mode=cv.StereoSGBM_MODE_HH,
)
for i in range(rect_frames.shape[1]):
    disparity = stereo.compute(rect_frames[0, i], rect_frames[1, i])
    cv.imwrite("/home/alphat/Desktop/disp/" + str(i) + ".jpg", disparity)
    # plt.figure()
    # plt.imshow(disparity, 'gray')
