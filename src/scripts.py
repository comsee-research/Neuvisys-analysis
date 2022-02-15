#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

from src.spiking_network.network.network_params import inhibition_disparity_params
from src.spiking_network.planning.network_planner import create_networks, launch_neuvisys_multi_pass
from src.spiking_network.network.neuvisys import SpikingNetwork

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis/src")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis/src")
    home = "/home/thomas/"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# %% Generate Spiking Network

exec_path = home + "neuvisys-dv/cmake-build-release/neuvisys-exe"
networks_path = home + "Bureau/"
event_path = home + "Videos/disparity/"

# disparity
events_0and4 = home + "Videos/disparity/0&4_disp.npz"
events_fulldisp = home + "Videos/disparity/-4&-2&0&2&4_disp.npz"
events_0 = home + "Videos/disparity/0_disp.npz"
events_4 = home + "Videos/disparity/4_disp.npz"
list_disparities = [home + "Videos/disparity/base_disp/" + str(disp) + ".npz" for disp in
                    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]

params = inhibition_disparity_params({'network_config': {'layerSizes': [[8, 8, 100], [2, 2, 16]],
                                                         'neuronSizes': [[10, 10, 1], [4, 4, 100]]},
                                      'simple_cell_config': {'VTHRESH': 20,
                                                             'ETA_INH': 15,
                                                             'STDP_LEARNING': 'excitatory',
                                                             'NORM_FACTOR': 4,
                                                             'ETA_LTP': 0.0000257,
                                                             'ETA_LTD': -0.000007, }})
create_networks(exec_path, networks_path, 1, params)

path = networks_path + "network_0/"

draw = np.random.randint(-5, 6, size=500)
for ind in draw:
    launch_neuvisys_multi_pass(exec_path, path + "configs/network_config.json", list_disparities[ind], 1)
