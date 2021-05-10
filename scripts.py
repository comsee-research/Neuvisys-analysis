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

network_path = home + "neuvisys-dv/configuration/network/"

import numpy as np
import matplotlib.pyplot as plt

from spiking_network.spiking_network import SpikingNetwork
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
    network_params,
    compute_disparity_0,
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

display_network([spinet], 0)


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=True, complex_cells=True, json_only=False)


#%% Load events

events = load_aedat4(home + "Desktop/Events/pavin-3-5.aedat4")


#%% Load rosbag and convert it to npdat

left_events = ros_to_npy(
    home + "Downloads/outdoor_night1_data.bag", topic="/davis/left/events"
)
right_events = ros_to_npy(
    home + "Downloads/outdoor_night1_data.bag", topic="/davis/right/events"
)


#%% Save aedat file as numpy npz file

write_npz(home + "Desktop/mvsec_car_night", (left_events, right_events))


#%% Load frames

frames = load_frames("/media/alphat/DisqueDur/0_Thesis/pavin.aedat4")


#%% Load network params of learned batch


network_path = "/home/thomas/Desktop/NetworkSuite/tau_rp/network_"
nb_networks = 10
ndf, sdf, cdf = network_params(network_path, nb_networks, trim_sim_val=True)


#%% Load various neuron informations

simpa_decay, compa_decay = load_array_param(spinet, "learning_decay")
simpa_spike, compa_spike = load_array_param(spinet, "count_spike")


#%% Plot cell response

pot_train = []
for i in range(1, 2):
    pot_train.append(np.array(spinet.complex_cells[i].potential_train))
y_train, x_train = np.array(pot_train)[0, :, 0], np.array(pot_train)[0, :, 1]


#%% Spike plots

spike_plots_simple_cells(spinet, 7639)
spike_plots_complex_cells(spinet, 100)


#%% Spike rate evolution

for i in range(10):
    spike_rate_evolution(spinet, i)


#%% Create Matlab weight.mat

basis = spinet.generate_weight_mat()


#%% Load and create gabor basis

spinet.generate_weight_images()
gabor_params_l = create_gabor_basis(spinet, "None", nb_ticks=8)
# gabor_params_l = create_gabor_basis(spinet, "left/", nb_ticks=8)
# gabor_params_r = create_gabor_basis(spinet, "right/", nb_ticks=8)


#%% Create plots for preferred orientations and directions

oris, oris_r = hists_preferred_orientations(spinet)
plot_preferred_orientations(spinet, oris, oris_r)

#%% direction and orientation selectivity

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


#%% Compute receptive field disparity

weights = spinet.get_weights("simple")
residuals, disparity = rf_matching(weights)
disparity[disparity >= 5] -= 10

compute_disparity_0(spinet, disparity, residuals, min_resi=0.5, max_resi=10)


#%% Stereo matching

disp_frames, disp_nb_frames = stereo_matching(
    "/home/alphat/Desktop/pavin_images/im1/",
    [10, 84, 158, 232],
    [20, 83, 146],
    range(0, 200),
)


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


#%% Launch training of multiple networks

n_networks = 1
exec_path = "/home/thomas/neuvisys-dv/build/neuvisys"
networks_path = "/home/thomas/Desktop/NetworkSuite/extreme/"
event_path = "/home/thomas/Desktop/NetworkSuite/shapes.npz"

generate_networks(networks_path, n_networks)
nb_iterations = 8

for i in range(0, n_networks):
    launch_neuvisys_multi_pass(
        exec_path,
        networks_path + "network_" + str(i) + "/configs/network_config.json",
        event_path,
        nb_iterations,
    )

    spinet = SpikingNetwork(networks_path + "network_" + str(i) + "/")
    display_network([spinet], 0)
    # basis = spinet.generate_weight_mat()


#%% Rectify and plot event images

rect_events = rectify_events((events[0].copy(), events[1].copy()), -5, -16, 5, 16)

for i in range(2):
    show_event_images(
        npaedat_to_np(rect_events[0]),
        100000,
        346,
        260,
        "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/",
        ([10, 84, 158, 232, 306], [20, 83, 146, 209]),
        "_"+str(i),
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

#%%

import rosbag
import seaborn as sns

bag_file = "/home/thomas/Downloads/outdoor_day1_gt.bag"
bag = rosbag.Bag(bag_file)

dt = np.dtype('<f4')

xs = [10, 148, 286]
ys = [10, 105, 200]
mat = [[[], [], []],
       [[], [], []],
       [[], [], []]]

cnt = 0
for topic, msg, t in bag.read_messages(topics=["/davis/left/depth_image_raw"]):
    # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
    im = np.frombuffer(msg.data, dtype=dt).reshape(260, 346)
    
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mat[i][j].append(im[y:y+40, x:x+40].flatten())

    # plt.imshow(im)
    # plt.show()
    
mat = np.array(mat)
mat = mat.reshape(mat.shape[:-2] + (-1,))

#%%

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
# fig.suptitle("Ground Truth Depth estimation per region", fontsize=30)
for i in range(len(xs)):
    for j in range(len(ys)):
        if j != 2:
            sns.histplot(mat[i, j][mat[i, j] < 100], ax=axes[j, i], stat="density", color="#2C363F")

for ax in axes.flat:
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    ax.set_ylabel('Density', fontsize=26)
    ax.set_xticks(np.arange(0, 110, 10))

axes[1, 1].set_xlabel('Depth (m)', fontsize=26)

plt.savefig("/home/thomas/Desktop/images/gt", bbox_inches="tight")

#%%

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
# fig.suptitle("Ground Truth Depth estimation per region", fontsize=30)
cnt = 0

for i in range(len(spinet.conf["L1XAnchor"])):
    for j in range(len(spinet.conf["L1YAnchor"])):
        mask_residual = (
            residuals[
                cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
            ]
            < 30
        ) & (
            residuals[
                cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"]
            ]
            > 0.5
        )
        if j != 2:
            sns.histplot(
                disparity[
                    cnt * spinet.conf["L1Depth"] : (cnt + 1) * spinet.conf["L1Depth"],
                    0,
                ][mask_residual],
                ax=axes[j, i],
                bins=[4.46, 5.575, 7.43, 11.15, 22.3, 70],
                stat="density",
                color="#2C363F"
            )
        cnt += 1

for i in range(len(xs)):
    for j in range(len(ys)):
        if j != 2:
            sns.histplot(mat[i, j][mat[i, j] < 70], ax=axes[j, i], stat="density")

for ax in axes.flat:
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_xticks(np.arange(0, 71, 10))

axes[1, 1].set_xlabel('Depth (m)', fontsize=24)

plt.savefig("/home/thomas/Desktop/images/test", bbox_inches="tight")