#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:58:28 2020

@author: thomas
"""

import os

from src.events.Events import Events

if os.path.exists("/home/alphat"):
    os.chdir("/home/alphat/neuvisys-analysis/src")
    home = "/home/alphat/"
else:
    os.chdir("/home/thomas/neuvisys-analysis/src")
    home = "/home/thomas/"

# %% Generate Spiking Network

events = Events("/home/thomas/Videos/DSEC/car_left.npz", "/home/thomas/Videos/DSEC/car_right.npz")
events.save_file("/home/thomas/Videos/DSEC/car")

# events.resize_events(147, 110, 346, 260)
# events.save_file("/home/thomas/Videos/DSEC/car_left")
# events.event_to_video(50, "/home/thomas/Videos/DSEC/car_left", 640, 480)
