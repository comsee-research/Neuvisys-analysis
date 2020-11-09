#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 04:47:53 2020

@author: alphat
"""

import numpy as np
from PIL import Image
import os
from natsort import natsorted
import matplotlib.pyplot as plt

os.chdir("/home/alphat/neuvisys-analysis")


def write_event(events, delta_B, thresh, frame_id, n_max, x, y, time_gap, polarity):
    moddiff = int(delta_B / thresh)
    if moddiff > n_max:
        nb_event = n_max
    else:
        nb_event = moddiff

    for e in range(nb_event):
        timestamp = int(((time_gap * (e + 1) * thresh) / delta_B) + time_gap * (frame_id))
        events.append([x, y, timestamp, polarity])
    
    return nb_event

def pixels_to_events(folder, update_method=1, log_threshold=20, map_threshold=0.4, swin=1, n_max=5, time_gap=1000, adapt_thresh_coef_shift=0.05):
    events = []
    threshold_map = np.full((346, 260), map_threshold)
    
    frames = natsorted(os.listdir(folder))
    
    reference = np.zeros((346, 260))
    if update_method != 3:
        reference = np.asarray(Image.open(folder+frames[0])).transpose(1, 0, 2)
        reference = 0.299 * reference[:, :, 0] + 0.587 * reference[:, :, 1] + 0.114 * reference[:, :, 2]
        np.log(reference, out=reference, where=reference > log_threshold)
    
    for frame_id, frame in enumerate(frames[1:]):
        frame = np.asarray(Image.open(folder+frame)).transpose(1, 0, 2)
        # luminance transformation
        frame = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        # log frame
        np.log(frame, out=frame, where=frame > log_threshold)
        
        for i in range(0, 346, swin):
            for j in range(0, 260, swin):
                ind_max = np.unravel_index(np.argmax(np.abs(frame[i:i+swin, j:j+swin] - reference[i:i+swin, j:j+swin])), (swin, swin))
                
                delta_B = frame[i+ind_max[0], j+ind_max[1]] - reference[i+ind_max[0], j+ind_max[1]]
                
                thresh = threshold_map[i+ind_max[0], j+ind_max[1]]
                if delta_B <= thresh and delta_B >= -thresh:
                    threshold_map[i, j] *= (1 - adapt_thresh_coef_shift)
                else:
                    if delta_B > thresh:
                        nb_event = write_event(events, delta_B, thresh, frame_id, n_max, i+ind_max[0], j+ind_max[1], time_gap, 1)
                        if update_method == 3:
                            reference[i, j] += nb_event * threshold_map[i, j]
                    elif delta_B < -thresh:
                        nb_event = write_event(events, -delta_B, thresh, frame_id, n_max, i+ind_max[0], j+ind_max[1], time_gap, 0)
                        if update_method == 3:
                            reference[i, j] -= nb_event * threshold_map[i, j]
                    
                    threshold_map[i, j] *= (1 + adapt_thresh_coef_shift)
                    
                    if update_method == 2:
                        reference[i, j] = frame[i, j]
                if update_method == 1:
                    reference[i:i+swin, j:j+swin] = frame[i:i+swin, j:j+swin]
    return np.array(events)

events = pixels_to_events("/home/alphat/Desktop/circles/")

times = np.diff(events[:, 2])
imgs = []
width = np.max(events[:, 0]) + 1
height = np.max(events[:, 1]) + 1

time = 0
img = np.zeros((height, width, 3))
for i in range(events.shape[0]-1):
    img[events[i, 1], events[i, 0], events[i, 3]] = 1
    time += times[i]
    if time > 1000:
        time = 0
        plt.figure()
        plt.imshow(img)
        img = np.zeros((height, width, 3))