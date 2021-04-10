#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis")
    home = "/home/thomas/"

network_path = home + "/neuvisys-dv/configuration/network/"

import json
import numpy as np
import matplotlib.pyplot as plt

from spiking_network.spiking_network import SpikingNetwork
from aedat_tools.aedat_tools import (
    stereo_matching,
    load_aedat4,
    show_event_images,
    write_npz,
    load_frames,
    npz_to_arr,
    npaedat_to_np,
    rectify_events,
    rectify_frames,
    remove_events,
    write_frames,
)
from spiking_network.display import (
    display_network,
    load_array_param,
    complex_cells_directions,
)
from event_statistics.frame_analysis import stereo_matching
from spiking_network.network_statistics.network_statistics import (
    compute_disparity,
    rf_matching,
    spike_plots_simple_cells,
    spike_plots_complex_cells,
    direction_norm_length,
    orientation_norm_length,
    direction_selectivity,
    orientation_selectivity,
    spike_rate_evolution,
)
from spiking_network.network_planning.planner import (
    generate_networks,
    launch_neuvisys_multi_pass,
    toggle_learning,
)
from spiking_network.gabor_fitting.gabbor_fitting import (
    create_gabor_basis,
    hists_preferred_orientations,
    plot_preferred_orientations,
)


#%% Generate Spiking Network

spinet = SpikingNetwork(network_path)


#%% Display weights

display_network([spinet], 1)


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=True, complex_cells=True, json_only=False)


#%% Load events

events = load_aedat4(home + "Desktop/Events/pavin-3-5.aedat4")


#%% Save aedat file as numpy npz file

write_npz(home + "Desktop/Events/pavin-3-1", events)


#%% Load frames

frames = load_frames("/media/alphat/DisqueDur/0_Thesis/pavin.aedat4")


#%% Launch training script

directory = "/home/thomas/neuvisys-dv/configuration/"
files = [
    "/home/thomas/Vidéos/driving_dataset/npy/mix_12.npy",
    "/home/thomas/Vidéos/driving_dataset/npy/mix_17.npy",
]
files = ["/home/thomas/Bureau/concat.npy"]

launch_spinet(directory, files, 1)


#%% Create Matlab weight.mat

basis = spinet.generate_weight_mat()


#%% Load and create gabor basis

spinet.generate_weight_images()
# gabor_params_l = create_gabor_basis(spinet, "None", nb_ticks=8)
gabor_params_l = create_gabor_basis(spinet, "left", nb_ticks=8)
gabor_params_r = create_gabor_basis(spinet, "right", nb_ticks=8)


#%% Compute receptive field disparity

weights = spinet.get_weights("simple")
residuals, disparity = rf_matching(weights)

disparity[disparity >= 5] -= 10
compute_disparity(
    spinet,
    disparity,
    gabor_params_l[4],
    (gabor_params_l[5] + gabor_params_r[5]) / 2,
    residuals,
    0,
    500,
    120.5,
)


#%% Stereo matching

# disp_frames, disp_nb_frames = stereo_matching("/home/alphat/Desktop/pavin_images/im3/", [10, 84, 158, 232], [20, 83, 146], range(900, 1100))
disp_frames, disp_nb_frames = stereo_matching(
    "/home/alphat/Desktop/pavin_images/im1/",
    [10, 84, 158, 232],
    [20, 83, 146],
    range(0, 200),
)


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

toggle_learning(spinet, True)

for i in range(3):
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_h.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_-90.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_180.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_hv.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_flip_v.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_0.npz",
        2,
    )
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/home/thomas/Videos/real_videos/npz/diverse_shapes/shapes_rot_90.npz",
        2,
    )

spinet = SpikingNetwork(network_path)
display_network([spinet], 0)


#%% Toggle learning

toggle_learning(spinet, False)


#%% Complex response to moving bars

sspikes = []
cspikes = []
rotations = np.array(
    [0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338]
)
for rot in rotations:
    launch_neuvisys_multi_pass(
        network_path + "configs/network_config.json",
        "/media/alphat/SSD Games/Thesis/videos/artificial_videos/lines/"
        + str(rot)
        + ".npz",
        5,
    )
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


#%% Launch training of multiple networks

n_networks = 1
networks_path = "/home/alphat/Desktop/Networks/"
event_path = "/home/alphat/Desktop/shapes.npz"
generate_networks(networks_path, n_networks)
nb_iterations = 20

for i in range(0, n_networks):
    launch_neuvisys_multi_pass(
        networks_path + "network_" + str(i) + "/configs/network_config.json",
        event_path,
        nb_iterations,
    )

    spinet = SpikingNetwork(networks_path + "network_" + str(i) + "/")
    display_network([spinet], 0)
    # basis = spinet.generate_weight_mat()


#%% Spike plots

spike_plots_simple_cells(spinet, 7986)
# spike_plots_complex_cells(spinet, 100)


#%% Spike rate evolution

for i in range(10):
    spike_rate_evolution(spinet, i)


#%% Rectify and Plot event images

rect_events = rectify_events((events[0].copy(), events[1].copy()), -5, -16, 5, 16)

show_event_images(
    npaedat_to_np(rect_events[0]),
    100000,
    346,
    260,
    "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/",
    ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
    "_l",
)
show_event_images(
    npaedat_to_np(rect_events[1]),
    100000,
    346,
    260,
    "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/",
    ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
    "_r",
)


#%% Plot frames

rect_frames = rectify_frames(frames, -4, 8, 4, -8)

write_frames(
    "/home/alphat/Desktop/im1/",
    rect_frames,
    ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
)


#%%

tss = [
    1615820915344885,
    1615820923944885,
    1615820925444885,
    1615820944844885,
    1615820947944885,
]
tse = [
    1615820916544885,
    1615820924344885,
    1615820925544885,
    1615820945244885,
    1615820948144885,
]

l_events, r_events = remove_events(rect_events, tss, tse)
