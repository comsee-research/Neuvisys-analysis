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

row = 5
col = 35

pdf = FPDF("P", "mm", (col*11, row*11))
pdf.add_page()

images = natsorted(os.listdir("/home/alphat/Desktop/temp_3/"))[::-1]

count = 0
for i in range(row):
    for j in range(col):
        pdf.image("/home/alphat/Desktop/temp_3/"+images[count], x=j*11, y=i*11, w=10, h=10)
        count += 1
        
pdf.output("/home/alphat/Desktop/images/test.pdf", "F")

#%%

rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
complex_cells_directions(spinet, rotations)

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
