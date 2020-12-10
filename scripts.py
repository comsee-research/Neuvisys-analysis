#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os
os.chdir("/home/alphat/neuvisys-analysis")

import json
import numpy as np
import matplotlib.pyplot as plt

from aedat_tools.aedat_tools import build_mixed_file, remove_blank_space, write_npdat, load_aedat4, convert_ros_to_aedat, concatenate_files
from graphical_interface.gui import launch_gui

from spiking_network.spiking_network import SpikingNetwork
from spiking_network.display import display_network, load_array_param, complex_cells_directions
from spiking_network.network_statistics.network_statistics import spike_plots, direction_norm_length, orientation_norm_length
from spiking_network.network_planning.planner import launch_spinet, launch_neuvisys_multi_pass, launch_neuvisys_stereo, toggle_learning
from spiking_network.gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations

# network_path = "/media/alphat/SSD Games/Thesis/Networks/network/"
network_path = "/home/alphat/neuvisys-dv/configuration/network/"

#%% Generate Spiking Network

spinet = SpikingNetwork(network_path)


#%% GUI

launch_gui(spinet)


#%% Display weights

display_network([spinet], 0)


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=True, complex_cells=True, json_only=False)


#%% Save aedat file as numpy array

events = load_aedat4("/media/alphat/SSD Games/Thesis/videos/left.aedat4")
write_npdat(events, "/media/alphat/SSD Games/Thesis/videos/left.npy")


#%% Build npdat file made of chunck of other files

path = "/home/thomas/Vidéos/driving_dataset/aedat4/"
files = [path+"campus_night_3.aedat4", path+"city_night_1.aedat4", path+"city_night_6.aedat4"]
chunk_size = 5000000

events = build_mixed_file(files, chunk_size)
write_npdat(events, "/home/thomas/Vidéos/driving_dataset/npy/mix_night_10.npy")
# write_aedat2_file(events, "/home/thomas/Bureau/concat.aedat", 346, 260)


#%% Remove blank space in aedat file

aedat4 = "/home/thomas/Vidéos/driving_dataset/aedat4/city_highway_night_16.aedat4"
aedat = "/home/thomas/Vidéos/driving_dataset/aedat/city_highway_night_16.aedat"

remove_blank_space(aedat4, aedat, 346, 260)


#%% Concatenate aedat4 in a single aedat file

inp = ["/home/thomas/Vidéos/samples/bars_vertical.aedat4", "/home/thomas/Vidéos/samples/bars_horizontal.aedat4"]
out = "/home/thomas/Bureau/concat.npy"

events = concatenate_files(inp)
write_npdat(events, out)
# write_aedat2_file(events, out, 346, 260)


#%% Launch training script

directory = "/home/thomas/neuvisys-dv/configuration/"
files = ["/home/thomas/Vidéos/driving_dataset/npy/mix_12.npy", "/home/thomas/Vidéos/driving_dataset/npy/mix_17.npy"]
files = ["/home/thomas/Bureau/concat.npy"]

launch_spinet(directory, files, 1)


#%% Create Matlab weight.mat

basis = spinet.generate_weight_mat()


#%% Load and create gabor basis

spinet.generate_weight_images()
create_gabor_basis(spinet, nb_ticks=8)


#%% Create plots for preferred orientations and directions

oris, oris_r = hists_preferred_orientations(spinet)
plot_preferred_orientations(spinet, oris, oris_r)


#%% Convert rosbag to aedat

convert_ros_to_aedat("/home/thomas/Bureau/flash_135.bag", "/home/thomas/Bureau/flash_135.aedat", 346, 260)


#%% Load various neuron informations

simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


#%% Plot cell response

pot_train = []
for i in range(1, 2):
    pot_train.append(np.array(spinet.complex_cells[i].potential_train))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]


#%% Launch

# launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shape_hovering.npy", 30)

for i in range(3):
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_flip_h.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_rot_-90.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_rot_180.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_flip_hv.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_flip_v.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_rot_0.npy", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/diverse_shapes/shapes_rot_90.npy", 2)

spinet = SpikingNetwork(network_path)
display_network([spinet], 0)
toggle_learning(spinet, False)

# Plot complex cells response directions

sspikes = []
cspikes = []

rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])

for rot in rotations:
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/artificial_videos/lines/"+str(rot)+".npy", 5)
    spinet = SpikingNetwork(network_path)
    sspikes.append(spinet.sspikes)
    cspikes.append(spinet.cspikes)
    
complex_cells_directions(spinet, rotations, cspikes)

    
#%%

angles = np.pi * rotations / 180

dirs = []
for i in range(144):
    dirs.append(direction_norm_length(spinet.directions[:, i], angles))
    
oris = []
for i in range(144):
    oris.append(orientation_norm_length(spinet.orientations[:, i], angles[0:8]))


#%% launch spinet with stereo setup

network_path = "/home/alphat/neuvisys-dv/configuration/network/"
launch_neuvisys_stereo(network_path+"configs/network_config.json",
                       "/media/alphat/SSD Games/Thesis/videos/artificial_videos/disparity_bars/disparity_bar_left.npy",
                       "/media/alphat/SSD Games/Thesis/videos/artificial_videos/disparity_bars/disparity_bar_right.npy", 200)
spinet = SpikingNetwork(network_path)
display_network([spinet], 0)


#%% Launch training of multiple networks

n_iter = 20
launch_spinet("/media/alphat/SSD Games/Thesis/configuration/", n_iter)
for i in range(0, n_iter):
    launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/configuration/network_"+str(i)+"/configs/network_config.json", "/media/alphat/SSD Games/Thesis/diverse_shapes/shape_hovering.npy", 25)
    
    spinet = SpikingNetwork("/media/alphat/SSD Games/Thesis/configuration/network_"+str(i)+"/")
    display_network([spinet], 0)
    basis = spinet.generate_weight_mat()


#%% Spike plots

spike_plots(spinet)