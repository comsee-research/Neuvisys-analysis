#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:47:24 2020

@author: alphat
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from aedat_tools.aedat_tools import txt_to_events, build_mixed_file, remove_blank_space, write_npdat, write_aedat2_file, load_aedat4, convert_ros_to_aedat, concatenate_files
from spiking_network import SpikingNetwork
from neuvisys_statistics.display_weights import display_network, load_array_param
from planning.planner import launch_spinet, launch_neuvisys_rotation, launch_neuvisys_multi_pass
from gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations
from gui import launch_gui
import pandas as pd


#%%

path = "/media/alphat/SSD Games/Thesis/configuration/network_"

a = []
for i in range(8):
    with open(path+str(i)+"/configs/pooling_neuron_config.json") as file:
        a.append(json.load(file))

df = pd.DataFrame(a)


#%% Get potential responses from a rotating stimulus

network_path = ""
spinet = SpikingNetwork(network_path)

rotation = np.array(np.arange(-180, 181, 22.5), dtype=np.int16)
nb_pass = 5

potentials = np.zeros((len(rotation), spinet.nb_complex_cells, nb_pass))
spikes = np.zeros((len(rotation), spinet.nb_complex_cells, nb_pass))

for i, rot in enumerate(rotation):
    for j in range(nb_pass):
        launch_neuvisys_rotation("/home/thomas/Vidéos/samples/npy/bars_horizontal_up_down.npy", rot)

        spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
        for k, neuron in enumerate(spinet.complex_cells):
            potentials[i, k, j] = np.mean(neuron.potential_train)
            spikes[i, k, j] = np.array(neuron.spike_train).size

potentials = np.mean(potentials, axis=-1)
spikes = np.mean(spikes, axis=-1)


#%% Get potential responses from a counterphase stimulus

rotation = [0, 45, 90, 135]
nb_pass = 5

potentials = np.zeros((len(rotation), spinet.nb_complex_cells, nb_pass))
spikes = np.zeros((len(rotation), spinet.nb_complex_cells, nb_pass))

for i, rot in enumerate(rotation):
    for j in range(nb_pass):
        launch_neuvisys_multi_pass("/home/thomas/Vidéos/samples/npy/flash_"+str(rot)+".npy", 1)

        spinet = SpikingNetwork("/home/thomas/neuvisys-dv/configuration/network/")
        for k, neuron in enumerate(spinet.complex_cells):
            potentials[i, k, j] = np.mean(neuron.potential_train)
            spikes[i, k, j] = np.array(neuron.spike_train).size

potentials = np.mean(potentials, axis=-1)
spikes = np.mean(spikes, axis=-1)


#%%

for i in range(spinet.nb_complex_cells):
    plt.figure()
    plt.title("Average spikes per orientation, Neuron: " + str(i))
    plt.plot(rotation, spikes[:, i])
    # plt.savefig("save/"+str(i))

# for i in range(spinet.nb_complex_cells):
#     y = np.mean(potentials[:, i, :, 0], axis=1)
#     plt.figure()
#     plt.title("mean cell response function of stimulus orientation")
#     plt.xticks(rotation[::2], rotation=45)
#     plt.plot(rotation, y)
#     plt.savefig("cell"+str(i))
    
# for i in range(spinet.nb_complex_cells):
#     y = np.max(train[:, i, :, 0], axis=1)
#     plt.figure()
#     plt.title("max cell response function of stimulus orientation")
#     plt.xticks(rotation[::2], rotation=45)
#     plt.plot(rotation, y)


#%% Create plots for preferred orientations and directions

oris, oris_r = hists_preferred_orientations(spinet)
plot_preferred_orientations(spinet, oris, oris_r)

OIs, DIs = [], []
for orie in oris:
    orie = orie[:-1]
    # dire = dire[:-1]
    OIs.append((np.max(orie) - orie[(np.argmax(orie)+4)%8]) / np.max(orie))
    # DIs.append((np.max(dire) - dire[(np.argmax(dire)+8)%16]) / np.max(dire))
    
OIsr, DIsr = [], []
for orie in oris_r:
    orie = orie[:-1]
    # dire = dire[:-1]
    OIsr.append((np.max(orie) - orie[(np.argmax(orie)+4)%8]) / np.max(orie))
    # DIsr.append((np.max(dire) - dire[(np.argmax(dire)+8)%16]) / np.max(dire))
   
            
#%%

l2 = []
l3 = []
for i in range(len(spinet.l1xanchor) * spinet.l1width):
    for j in range(len(spinet.l1yanchor) * spinet.l1height):
        l = []
        for k in range(spinet.l1depth):
            l.append(len(spikes[(i, j, k)]))
        l2.append(np.mean(l))
        l3.append(np.std(l))
        
