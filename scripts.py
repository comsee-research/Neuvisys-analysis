#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""
import os
import numpy as np
import json
import matplotlib.pyplot as plt

os.chdir("/home/thomas/neuvisys-analysis")

from aedat_tools.aedat_tools import build_mixed_file, remove_blank_space, write_npdat, write_aedat2_file, load_aedat4, convert_ros_to_aedat, concatenate_files
from spiking_network import SpikingNetwork
from neuvisys_statistics.display_weights import display_network, load_array_param
from planning.planner import launch_spinet, launch_neuvisys_rotation, launch_neuvisys_multi_pass
from gabor_fitting.gabbor_fitting import create_gabor_basis
from gui import launch_gui

#%% GUI

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
launch_gui(spinet)


#%% Display weights

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
img = display_network([spinet], 1)


#%% Save aedat file as numpy array

events = load_aedat4("/home/thomas/Vidéos/samples/shape_hovering.aedat4")
write_npdat(events, "/home/thomas/Vidéos/samples/npy/shape_hovering.npy")


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

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
basis = spinet.generate_weight_mat()


#%% Load and create gabor basis

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
spinet.generate_weight_images()
create_gabor_basis(spinet, bins=15)


#%% Convert rosbag to aedat

convert_ros_to_aedat("/home/thomas/Bureau/out.bag", "/home/thomas/Bureau/test.aedat", 346, 260)


#%% //!!!\\ Delete weights network

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
spinet.clean_network(simple_cells=0, complex_cells=1)


#%% Load various neuron informations

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


#%%

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
pot_train = []
for i in range(1):
    pot_train.append(np.array(spinet.complex_cells[i].params["potential_train"]))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]

    
#%%

for file in os.listdir("/home/thomas/neuvisys-dv/configuration/network/weights/simple_cells/"):
    if file.endswith(".json"):
        os.remove("/home/thomas/neuvisys-dv/configuration/network/weights/simple_cells/"+file)

        
#%% Get potential responses from a rotating stimulus

rotation = list(range(-180, 181, 10))
train = []
for rot in rotation:
    launch_neuvisys_rotation("/home/thomas/Vidéos/samples/npy/bars_horizontal_up_down.npy", rot)

    spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
    pot_train = []
    for neuron in spinet.complex_cells:
        pot_train.append(np.array(neuron.params["potential_train"]))
    # y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]
    train.append(pot_train)

train = np.array(train)

for i in range(spinet.nb_complex_cells):
    y = np.mean(train[:, i, :, 0], axis=1)
    plt.figure()
    plt.title("mean cell response function of stimulus orientation")
    plt.xticks(rotation[::2], rotation=45)
    plt.axvline(-90)
    plt.axvline(0)
    plt.axvline(90)
    plt.plot(rotation, y)
    plt.savefig("cell"+str(i))
    
# for i in range(spinet.nb_complex_cells):
#     y = np.max(train[:, i, :, 0], axis=1)
#     plt.figure()
#     plt.title("max cell response function of stimulus orientation")
#     plt.xticks(rotation[::2], rotation=45)
#     plt.axvline(-90)
#     plt.axvline(0)
#     plt.axvline(90)
#     plt.plot(rotation, y)


#%% Launch

launch_neuvisys_multi_pass("/home/thomas/Vidéos/samples/npy/shape_hovering.npy", 20)