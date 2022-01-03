#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

from src.spiking_network.analysis.network_display import display_network

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis/src")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis/src")
    home = "/home/thomas/"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.spiking_network.network.neuvisys import SpikingNetwork

# network_path = home + "neuvisys-dv/configuration/network/"
network_path = "/home/thomas/Downloads/network_ref_vh/"


# %% Generate Spiking Network

spinet = SpikingNetwork(network_path)

