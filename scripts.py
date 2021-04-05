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

network_path = home+"/neuvisys-dv/configuration/network/"

import json
import numpy as np
import matplotlib.pyplot as plt

from aedat_tools.aedat_tools import stereo_matching, load_aedat4, show_event_images, write_npz, load_frames, npz_to_arr, npaedat_to_np, rectify_events, rectify_frames, remove_events, write_frames

from spiking_network.spiking_network import SpikingNetwork
from spiking_network.display import display_network, load_array_param, complex_cells_directions
from spiking_network.network_statistics.network_statistics import compute_disparity, rf_matching, spike_plots_simple_cells, spike_plots_complex_cells, direction_norm_length, orientation_norm_length, direction_selectivity, orientation_selectivity
from spiking_network.network_planning.planner import launch_spinet, launch_neuvisys_multi_pass, launch_neuvisys_stereo, toggle_learning
from spiking_network.gabor_fitting.gabbor_fitting import create_gabor_basis, hists_preferred_orientations, plot_preferred_orientations


#%% Generate Spiking Network

spinet = SpikingNetwork(network_path)


#%% Display weights

display_network([spinet], 1)


#%% //!!!\\ Delete weights network

spinet.clean_network(simple_cells=True, complex_cells=True, json_only=False)


#%% Load events

events = load_aedat4(home+"Desktop/Events/pavin-3-5.aedat4")


#%% Save aedat file as numpy npz file

write_npz(home+"Desktop/Events/pavin-3-1", events)


#%% Load frames

frames = load_frames("/media/alphat/DisqueDur/0_Thesis/pavin.aedat4")


#%% Launch training script

directory = "/home/thomas/neuvisys-dv/configuration/"
files = ["/home/thomas/Vidéos/driving_dataset/npy/mix_12.npy", "/home/thomas/Vidéos/driving_dataset/npy/mix_17.npy"]
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
compute_disparity(spinet, disparity, gabor_params_l[4], (gabor_params_l[5] + gabor_params_r[5]) / 2, residuals, 0, 500, 120.5)


#%% Stereo matching

# disp_frames, disp_nb_frames = stereo_matching("/home/alphat/Desktop/pavin_images/im3/", [10, 84, 158, 232], [20, 83, 146], range(900, 1100))
disp_frames, disp_nb_frames = stereo_matching("/home/alphat/Desktop/pavin_images/im1/", [10, 84, 158, 232], [20, 83, 146], range(0, 200))


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
    launch_neuvisys_multi_pass(network_path+"configs/network_config.json", "/media/alphat/SSD Games/Thesis/videos/artificial_videos/lines/"+str(rot)+".npz", 5)
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


#%% Rectify and Plot event images

rect_events = rectify_events((events[0].copy(), events[1].copy()), -5, -16, 5, 16)

show_event_images(npaedat_to_np(rect_events[0]), 100000, 346, 260, "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/", ([10, 84, 158, 232, 306], [20, 83, 146, 209]), "_l")
show_event_images(npaedat_to_np(rect_events[1]), 100000, 346, 260, "/media/alphat/DisqueDur/0_Thesis/short_pavin2_img/", ([10, 84, 158, 232, 306], [20, 83, 146, 209]), "_r")


#%% Plot frames

rect_frames = rectify_frames(frames, -4, 8, 4, -8)

write_frames("/home/alphat/Desktop/im1/", rect_frames, ([10, 84, 158, 232, 306], [20, 83, 146, 209]))


#%%

tss = [1615820915344885, 1615820923944885, 1615820925444885, 1615820944844885, 1615820947944885]
tse = [1615820916544885, 1615820924344885, 1615820925544885, 1615820945244885, 1615820948144885]

l_events, r_events = remove_events(rect_events, tss, tse)


#%%

import cv2 as cv
import seaborn as sns

folder, xs, ys = "/home/alphat/Desktop/Bundle/pavin_images/im1_rect/", [10, 84, 158], [20]

mat = np.zeros((346, 260))
ind = 1
for x in xs:
    for y in ys:
        mat[x:x+30, y:y+30] = ind
        ind += 1
        
vec = {}
for i in range(21):
    vec[i] = []

for i in np.arange(0, 824):
    lframe = cv.imread(folder+"img"+str(i)+"_left.jpg")
    rframe = cv.imread(folder+"img"+str(i)+"_right.jpg")
    
    orb = cv.ORB_create(nfeatures=10000)
    
    kp_left, ds_left = orb.detectAndCompute(lframe, None)
    kp_right, ds_right = orb.detectAndCompute(rframe, None)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ds_left, ds_right)
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    for match in matches:
        lp = kp_left[match.queryIdx].pt
        rp = kp_right[match.trainIdx].pt
        
        x_shift = lp[0] - rp[0]
        y_shift = lp[1] - rp[1]
        # print("{:.1f}, {:.1f}".format(*lp), "|", "{:.1f}, {:.1f}".format(*rp), "->", "{:.2f}".format(x_shift), "|", "{:.2f}".format(y_shift))
        
        if np.abs(x_shift) < 5 and np.abs(y_shift) < 5:
            vec[mat[int(np.round((lp[0]))), int(np.round(lp[1]))]].append([x_shift, y_shift])
            
        # imgmatching = cv.drawMatches(lframe, kp_left, rframe, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(imgmatching)

fig, axes = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(16, 12))
fin = np.zeros((len(ys), len(xs), 2))
nb_fin = np.zeros((len(ys), len(xs)))
ind = 1
for i in range(len(xs)):
    for j in range(len(ys)):
        # axes[i].set_title("mean: " + "{:.2f}".format(np.mean(vec[ind], axis=0)[0]), fontsize=20)
        axes[i].hist(np.array(vec[ind])[:, 0], bins=np.arange(-5.5, 6.5), density=True, alpha=0.85, color="#16697A")
        plt.setp(axes[i].get_xticklabels(), fontsize=16)
        plt.setp(axes[i].get_yticklabels(), fontsize=16)
        axes[i].set_xticks(np.arange(-5, 6))
        fin[j, i] = np.mean(vec[ind], axis=0)
        nb_fin[j, i] = len(vec[ind])
        ind += 1

theta = gabor_params_l[4]
error = (gabor_params_l[5] + gabor_params_r[5]) / 2
epsi_theta, epsi_error, epsi_residual = 0, 500, 120.5
cnt = 0
for i in range(5):
    for j in range(4):
        mask_residual = residuals[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_residual
        mask_theta = (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] > np.pi/2+epsi_theta) | (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < np.pi/2-epsi_theta)
        mask_error = error[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_error
        if i < 3 and j < 1:
            # axes[i].set_title("mean: " + "{:.2f}".format(np.mean(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 0][mask_theta & mask_error])), fontsize=20)
            axes[i].hist(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 0][mask_theta & mask_error & mask_residual], bins=np.arange(-5.5, 6.5), density=True, alpha=0.7, color="#FFA62B")
        cnt += 1
axes[1].set_xlabel("Disparity (px)", fontsize=18)
axes[0].set_ylabel("Density", fontsize=18)
plt.savefig("/home/alphat/Desktop/images/orb_disparity.png", bbox_inches="tight")