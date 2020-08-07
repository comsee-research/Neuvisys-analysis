#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""
import os

os.chdir("/home/thomas/neuvisys-analysis")

from aedat_tools.aedat_tools import build_mixed_file, remove_blank_space, write_npdat, load_aedat4, convert_ros_to_aedat, concatenate_files
from spiking_network import SpikingNetwork
from neuvisys_statistics.display_weights import display_network
from planning.planner import launch_spinet
from gabor_fitting.gabbor_fitting import create_gabor_basis


#%% Display weights

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
display_network([spinet], 1)

#%% Save aedat file as numpy array

events = load_aedat4("/home/thomas/Vidéos/driving_dataset/aedat4/city_day_6.aedat4")
write_npdat(events, "/home/thomas/Vidéos/driving_dataset/npy/city_day_6.npy")

#%% Build aedat file made of chunck of other files

path = "/home/thomas/Vidéos/driving_dataset/aedat4/"
files = [path+"campus_night_3.aedat4", path+"city_night_1.aedat4", path+"city_night_6.aedat4"]
chunk_size = 5000000

events = build_mixed_file(files, chunk_size)
write_npdat(events, "/home/thomas/Vidéos/driving_dataset/npy/mix_night_10.npy")

#%% Remove blank space in aedat file

aedat4 = "/home/thomas/Vidéos/driving_dataset/aedat4/city_highway_night_16.aedat4"
aedat = "/home/thomas/Vidéos/driving_dataset/aedat/city_highway_night_16.aedat"

remove_blank_space(aedat4, aedat, 346, 260)

#%% Concatenate aedat4 files

aedat4_files = ["/home/thomas/Vidéos/samples/horizontal_bars.aedat4", "/home/thomas/Vidéos/samples/vertical_bars.aedat4"]
aedat = "/home/thomas/Bureau/concat.aedat"

concatenate_files(aedat4_files, aedat, 346, 260)

#%% Launch training script

directory = "/home/thomas/neuvisys-dv/configuration/"
files = ["/home/thomas/Vidéos/driving_dataset/npy/mix_12.npy", "/home/thomas/Vidéos/driving_dataset/npy/mix_17.npy"]

launch_spinet(directory, files, 1)

#%% Create Matlab weight.mat

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network_0/")
spinet.generate_weight_mat("/home/thomas/neuvisys-dv/configuration/network_0/gabors/weights.mat")

#%% Load and create gabor basis

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
create_gabor_basis(spinet, "/home/thomas/neuvisys-dv/configuration/network/gabors/")

#%% Convert rosbag to aedat

convert_ros_to_aedat("/home/thomas/Bureau/out.bag", "/home/thomas/Bureau/test.aedat", 346, 260)

#%% //!!!\\ Delete weights network

spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
spinet.clean_network()
