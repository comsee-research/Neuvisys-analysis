#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis/")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis/")
    home = "/home/thomas/"

import shutil
import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt

# %% Generate Spiking Network

# exec_path = home + "neuvisys-dv/cmake-build-release/neuvisys-exe"
# networks_path = home + "Desktop/Experiment/"
# event_path = home + "Videos/disparity/"

# create_networks(exec_path, networks_path, 1, inhibition_orientation_9regions())

# path = networks_path + "network_0/"

# for i in range(100):
#     launch_neuvisys_multi_pass(exec_path, path + "configs/network_config.json", "/home/thomas/Videos/DSEC/car_left.npz", 1)

# events = Events("/home/thomas/Videos/DSEC/car_left.npz", "/home/thomas/Videos/DSEC/car_right.npz")
# events.save_file("/home/thomas/Videos/DSEC/car")

# events.resize_events(147, 110, 346, 260)
# events.save_file("/home/thomas/Videos/DSEC/car_left")
# events.event_to_video(50, "/home/thomas/Videos/DSEC/car_left", 640, 480)
