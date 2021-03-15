#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:47:24 2020

@author: alphat
"""

import os
os.chdir("/home/alphat/neuvisys-analysis")

import json
import numpy as np
import matplotlib.pyplot as plt

from aedat_tools.aedat_tools import build_mixed_file, remove_blank_space, write_npdat, load_aedat4, load_aedat4_stereo, convert_ros_to_aedat, concatenate_files
from graphical_interface.gui import launch_gui

from spiking_network.spiking_network import SpikingNetwork
from spiking_network.display import display_network, load_array_param, complex_cells_directions
from spiking_network.network_statistics.network_statistics import spike_plots, direction_norm_length, orientation_norm_length, direction_selectivity, orientation_selectivity
from spiking_network.network_planning.planner import launch_spinet, launch_neuvisys_multi_pass, launch_neuvisys_stereo, toggle_learning
from spiking_network.gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations

import pandas as pd


#%%

path = "/media/alphat/SSD Games/Thesis/configuration/network_"

a = []
for i in range(8):
    with open(path+str(i)+"/configs/pooling_neuron_config.json") as file:
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
from fpdf import FPDF
from natsort import natsorted
import random

row = 6
col = 6
im_size = (20, 20)
im_size_pad = (22, 22)

pdf = FPDF("P", "mm", (col*im_size_pad[0], row*im_size_pad[1]))
pdf.add_page()

path = "/home/alphat/neuvisys-dv/configuration/NETWORKS/complex_selectivity/figures/complex_directions/"

images = natsorted(os.listdir(path))#[::-1]
random.shuffle(images)

count = 0
for i in range(row):
    for j in range(col):
        pdf.image(path+images[count], x=j*im_size_pad[0], y=i*im_size_pad[1], w=im_size[0], h=im_size[1])
        count += 1
        
pdf.output("/home/alphat/Desktop/images/directions.pdf", "F")

#%%

rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
dir_vec, ori_vec = complex_cells_directions(spinet, rotations)

angles = np.pi * rotations / 180

dirs = []
dis = []
for i in range(144):
    dirs.append(direction_norm_length(spinet.directions[:, i], angles))
    dis.append(direction_selectivity(spinet.directions[:, i]))
oris = []
ois = []
for i in range(144):
    oris.append(orientation_norm_length(spinet.orientations[:, i], angles[0:8]))
    ois.append(orientation_selectivity(spinet.orientations[:, i]))
    
#%%

fig, ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

color1 = dict(color="#2C363F")
color2 = dict(color="#9E7B9B")

ax.set_ylabel("normalized vector length")
ax.boxplot(np.abs(des), positions=[0], labels=["direction exponential"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(oes), positions=[1], labels=["orientation exponential"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dls), positions=[2], labels=["direction linear"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(ols), positions=[3], labels=["orientation linear"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dsr), positions=[4], labels=["direction step left"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(osr), positions=[5], labels=["orientation step left"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dsl), positions=[6], labels=["direction step right"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(osl), positions=[7], labels=["orientation step right"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dss), positions=[8], labels=["direction step sym"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(oss), positions=[9], labels=["orientation step sym"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
plt.show()

#%%

events[0]["x"] -= 4
events[1]["x"] += 4
events[0]["y"] += 9
events[1]["y"] -= 8

l_events = events[0][(events[0]["y"] < 260) & (events[0]["x"] >= 0)]
r_events = events[1][(events[1]["y"] >= 0) & (events[1]["x"] < 346)]


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
lin[(t > -tau_ltd) & (t <= 0)] = -eta_ltd * -t[(t > -tau_ltd) & (t <= 0)][::-1] / np.max(-t[(t > -tau_ltd) & (t <= 0)][::-1])
lin[(t >= 0) & (t < tau_ltd)] = eta_ltp * t[(t >= 0) & (t < tau_ltd)][::-1] / np.max(t[(t >= 0) & (t < tau_ltd)][::-1])

exp = np.concatenate((eta_ltd * np.exp(t[t < 0] / tau_ltd), eta_ltp * np.exp(-t[t >= 0] / tau_ltp)))


fig, axs = plt.subplots(3, 1)

for ax in axs.flat:
    ax.grid(alpha=.2, linestyle='--')
    ax.axhline(0, alpha=0.3, linestyle='-', color='k')
    ax.axvline(0, alpha=0.3, linestyle='-', color='k')
    ax.xlim = [-0.5, 0.5]
    ax.set_ylabel("Synaptic change (mV)")
    ax.label_outer()

axs[1].set_xlabel("Spike timing (ms)")

axs[0].plot(t, step, color='k')
axs[0].set_title("Step")
axs[1].plot(t, lin, color='k')
axs[1].set_title("Linear")
axs[2].plot(t, exp, color='k')
axs[2].set_title("Exponential")