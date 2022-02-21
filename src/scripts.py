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
from src.events.Events import Events

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

events = Events("/home/thomas/Bureau/DSEC/interlaken_00_c_events_left/events.h5")

events.event_to_video(50, "/home/thomas/Bureau/test", 640, 480)