#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os
os.chdir("/home/thomas/neuvisys-analysis")

import json
import numpy as np
import matplotlib.pyplot as plt

from aedat_tools.aedat_tools import load_aedat4, load_aedat4_stereo, show_event_images, write_npz
from graphical_interface.gui import launch_gui

from spiking_network.spiking_network import SpikingNetwork
from spiking_network.display import display_network, load_array_param, complex_cells_directions
from spiking_network.network_statistics.network_statistics import spike_plots_simple_cells, spike_plots_complex_cells, direction_norm_length, orientation_norm_length, direction_selectivity, orientation_selectivity
from spiking_network.network_planning.planner import launch_spinet, launch_neuvisys_multi_pass, launch_neuvisys_stereo, toggle_learning
from spiking_network.gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations


network_path = "/home/thomas/neuvisys-dv/configuration/network/"

#%% Generate Spiking Network

spinet = SpikingNetwork(network_path)


#%% GUI

launch_gui(spinet)


#%% Display weights

display_network([spinet], 1)


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=True, complex_cells=True, json_only=False)


#%% Save aedat file as numpy npz file

events = load_aedat4("/home/thomas/Vidéos/real_videos/aedat4/diverse_shapes/shapes_rot_180.aedat4")
write_npz("/home/thomas/Vidéos/real_videos/npz/diverse_shapes/shapes_rot_180.npz", events)


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


#%% Load various neuron informations

simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


#%% Plot cell response

pot_train = []
for i in range(1, 2):
    pot_train.append(np.array(spinet.complex_cells[i].potential_train))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]


#%% Launch

for i in range(3):
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_h.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_-90.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_180.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_hv.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_v.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_0.npz", 2)
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_90.npz", 2)

spinet = SpikingNetwork(network_path)
display_network([spinet], 0)

#%% Toggle learning

toggle_learning(spinet, False)

#%% Complex response to moving bars

sspikes = []
cspikes = []
rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
for rot in rotations:
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/artificial_videos/lines/"+str(rot)+".npy", 5)
    spinet = SpikingNetwork(network_path)
    sspikes.append(spinet.sspikes)
    cspikes.append(spinet.cspikes)
spinet.save_complex_directions(cspikes, rotations)


#%%

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


#%% launch spinet with stereo setup

network_path = "/home/alphat/neuvisys-dv/configuration/network/"
launch_neuvisys_stereo(network_path+"configs/network_config.json",
                       "/home/alphat/Desktop/l_events.npz",
                       "/home/alphat/Desktop/r_events.npz", 1)
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

spike_plots_simple_cells(spinet, 7639)
# spike_plots_complex_cells(spinet, 100)


#%% Plot event images

# show_event_images(l_events, 1000000)