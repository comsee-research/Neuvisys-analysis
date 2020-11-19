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
from aedat_tools.aedat_tools import txt_to_events, build_mixed_file, remove_blank_space, write_npdat, write_aedat2_file, load_aedat4, convert_ros_to_aedat, concatenate_files
from spiking_network import SpikingNetwork
from neuvisys_statistics.display_weights import display_network, load_array_param
from planning.planner import launch_spinet, launch_neuvisys_rotation, launch_neuvisys_multi_pass
from gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations
from gui import launch_gui

network_path = "/media/alphat/SSD Games/Thesis/network/"

#%% Generate Spiking Network

spinet = SpikingNetwork(network_path)


#%% GUI

launch_gui(spinet)


#%% Display weights

display_network([spinet], 1)


#%% Save aedat file as numpy array

events = load_aedat4("/home/alphat/Desktop/diverse_shapes/shapes_rot_-90.aedat4")
write_npdat(events, "/home/alphat/Desktop/diverse_shapes/shapes_rot_-90.npy")


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


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=1, complex_cells=1)


#%% Load various neuron informations

simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


#%% Plot cell response

pot_train = []
for i in range(1, 2):
    pot_train.append(np.array(spinet.complex_cells[i].potential_train))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]


#%% Launch

# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_flip_h.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_rot_-90.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_rot_180.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_flip_hv.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_flip_v.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_rot_0.npy", 1)
# launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/diverse_shapes/shapes_rot_90.npy", 1)

launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/Networks/network/configs/network_config.json", "/media/alphat/SSD Games/Thesis/diverse_shapes/shape_hovering.npy", 5)
spinet = SpikingNetwork(network_path)


#%% Launch training of multiple networks

n_iter = 20
launch_spinet("/media/alphat/SSD Games/Thesis/configuration/", n_iter)
for i in range(0, n_iter):
    launch_neuvisys_multi_pass("/media/alphat/SSD Games/Thesis/configuration/network_"+str(i)+"/configs/network_config.json", "/media/alphat/SSD Games/Thesis/diverse_shapes/shape_hovering.npy", 25)
    
    spinet = SpikingNetwork("/media/alphat/SSD Games/Thesis/configuration/network_"+str(i)+"/")
    display_network([spinet], 0)
    basis = spinet.generate_weight_mat()


#%% Remove json files only

for file in os.listdir("/home/thomas/neuvisys-dv/configuration/network/weights/simple_cells/"):
    if file.endswith(".json"):
        os.remove("/home/thomas/neuvisys-dv/configuration/network/weights/simple_cells/"+file)
        
#%%

network_path = "/media/alphat/SSD Games/Thesis/configuration/network_7/"
spinet = SpikingNetwork(network_path)
spinet.generate_weight_images()
create_gabor_basis(spinet, nb_ticks=8)
oris, oris_r = hists_preferred_orientations(spinet)
plot_preferred_orientations(spinet, oris, oris_r)