#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from spiking_network.spiking_network import SpikingNetwork, clean_network
from aedat_tools.aedat_tools import (
    load_aedat4,
    show_event_images,
    write_npz,
    load_frames,
    npaedat_to_np,
    ros_to_npy,
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
    compute_disparity_0,
    rf_matching,
    spike_plots_simple_cells,
    spike_plots_complex_cells,
    direction_norm_length,
    orientation_norm_length,
    direction_selectivity,
    orientation_selectivity,
    spike_rate_evolution,
    update_dataframe,
)
from spiking_network.network_planning.planner import (
    generate_networks,
    generate_network_files,
    launch_neuvisys_multi_pass,
    toggle_learning,
    divide_visual_field,
)
from spiking_network.gabor_fitting.gabbor_fitting import (
    create_gabor_basis,
    hists_preferred_orientations,
    plot_preferred_orientations,
)
from spiking_network.network_statistics.parameter_analysis import (
    correlation_matrix,
    scatter_mat,
)

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis")
    home = "/home/thomas/"

network_path = home + "neuvisys-dv/configuration/network/"


# %% Generate Spiking Network

spinet = SpikingNetwork(network_path)


# %% Display weights

display_network([spinet])


# %% //!!!\\ Delete weights network

clean_network(
    network_path,
    simple_cells=True,
    complex_cells=True,
    motor_cells=True,
    json_only=False,
)


# %% Generate network

generate_network_files(home + "neuvisys-dv/configuration/")


# %% Load events

events = load_aedat4(home + "Desktop/shape_hovering_fast.aedat4")


# %% Load rosbag and convert it to npdat

left_events = ros_to_npy(
    home + "Downloads/outdoor_night1_data.bag", topic="/davis/left/events"
)
right_events = ros_to_npy(
    home + "Downloads/outdoor_night1_data.bag", topic="/davis/right/events"
)


# %% Save aedat file as numpy npz file

write_npz(home + "Desktop/shapes", events)  # mono
# write_npz(home + "Desktop/shapes", (left_events, right_events)) # stereo


# %% Load frames

frames = load_frames("/media/alphat/DisqueDur/0_Thesis/pavin.aedat4")


# %% Load various neuron informations

simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


# %% Plot cell response

pot_train = []
for i in range(1, 2):
    pot_train.append(np.array(spinet.complex_cells[i].potential_train))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]


# %% Spike plots

spike_plots_simple_cells(spinet, 7639)
spike_plots_complex_cells(spinet, 100)


# %% Spike rate evolution

for i in range(10):
    spike_rate_evolution(spinet, i)


# %% Create Matlab weight.mat

basis = spinet.generate_weight_mat()


# %% Load and create gabor basis

spinet.generate_weight_images()
gabor_params_l = create_gabor_basis(spinet, "None", nb_ticks=8)
# gabor_params_l = create_gabor_basis(spinet, "left/", nb_ticks=8)
# gabor_params_r = create_gabor_basis(spinet, "right/", nb_ticks=8)


# %% Create plots for preferred orientations and directions

oris, oris_r = hists_preferred_orientations(spinet)
plot_preferred_orientations(spinet, oris, oris_r)


# %% direction and orientation selectivity

rotations = np.array(
    [0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338]
)
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


# %% Compute receptive field disparity

weights = spinet.get_weights("simple")
residuals, disparity = rf_matching(weights)
disparity[disparity >= 5] -= 10

compute_disparity_0(spinet, disparity, residuals, min_resi=0.5, max_resi=10)


# %% Stereo matching

disp_frames, disp_nb_frames = stereo_matching(
    "/home/alphat/Desktop/pavin_images/im1/",
    [10, 84, 158, 232],
    [20, 83, 146],
    range(0, 200),
)


# %% Toggle learning

toggle_learning(spinet, False)


# %% Complex response to moving bars

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


# %% Launch training of multiple networks

n_networks = 100
exec_path = home + "neuvisys-dv/build/neuvisys-exe"
networks_path = home + "Desktop/test/"
event_path = home + "Desktop/shapes.npz"

# params = {"neuron_params": {"TARGET_SPIKE_RATE": [0.1, 0.2, 0.3]}}
params = {}

generate_networks(networks_path, params, n_networks)
nb_iterations = 5

df = []

for i in range(0, n_networks):
    launch_neuvisys_multi_pass(
        exec_path,
        networks_path + "network_" + str(i) + "/configs/network_config.json",
        event_path,
        nb_iterations,
    )

    spinet = SpikingNetwork(networks_path + "network_" + str(i) + "/")
    display_network([spinet], 0)
    update_dataframe(df, spinet)

df = pd.DataFrame(df)
df.to_csv(home + "Desktop/test/df")


# %% Plot correlations

# df = pd.read_csv(home + "Desktop/test/df")
temp = df.drop(columns=["TRACKING", "SYNAPSE_DELAY", "STDP_LEARNING", "MIN_THRESH"])

correlation_matrix(temp.iloc[:, 25:])
scatter_mat(temp.iloc[:, 25:])


# %% Rectify and plot event images

rect_events = rectify_events((events[0].copy(), events[1].copy()), -5, -16, 5, 16)

for i in range(2):
    show_event_images(
        npaedat_to_np(rect_events[0]),
        100000,
        346,
        260,
        "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/",
        ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
        "_" + str(i),
    )


# %% Plot frames

rect_frames = rectify_frames(frames, -4, 8, 4, -8)

write_frames(
    "/home/alphat/Desktop/im1/",
    rect_frames,
    ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
)


# %% remove events from specific intervals

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


# %% divide visual fields

X, Y = divide_visual_field(5, 1, 40, 40)
