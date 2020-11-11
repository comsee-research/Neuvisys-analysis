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

class Pix2Eve:
    """Transform frames into an event stream"""
    
    def __init__(self, folder, time_gap, log_threshold=20, map_threshold=0.4, swin=1, n_max=5, adapt_thresh_coef_shift=0.05):
        self.folder = folder
        self.time_gap = time_gap
        # self.update_method = update_method
        self.log_threshold = log_threshold
        self.map_threshold = map_threshold
        # self.swin = swin
        self.n_max = n_max
        self.adapt_thresh_coef_shift = adapt_thresh_coef_shift
        
    def write_event(self, events, delta_B, thresh, frame_id, x, y, polarity):
        moddiff = int(delta_B / thresh)
        if moddiff > self.n_max:
            nb_event = self.n_max
        else:
            nb_event = moddiff
    
        for e in range(nb_event):
            timestamp = int(((self.time_gap * (e + 1) * thresh) / delta_B) + self.time_gap * frame_id)
            events.append([x, y, timestamp, polarity])
        
        return nb_event
        
    def convert_frame(self, frame):
        frame = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        np.log(frame, out=frame, where=frame > self.log_threshold)
        return frame
    
    def frame_to_events(self, frame_id, frame, reference, threshold_map, events):
        delta = frame - reference
        
        for i, j in zip(*np.nonzero(delta > threshold_map)):            
            self.write_event(events, delta[i, j], threshold_map[i, j], frame_id, i, j, 1)
        
        for i, j in zip(*np.nonzero(delta < -threshold_map)):
            self.write_event(events, -delta[i, j], threshold_map[i, j], frame_id, i, j, 0)
        
        threshold_map[(delta > threshold_map) | (delta < -threshold_map)] *= (1 + self.adapt_thresh_coef_shift)
        threshold_map[(delta <= threshold_map) & (delta >= -threshold_map)] *= (1 - self.adapt_thresh_coef_shift)
    
    def run(self):
        events = []
        threshold_map = np.full((346, 260), self.map_threshold)
        
        frames = natsorted(os.listdir(self.folder))
        reference = self.convert_frame(np.asarray(Image.open(self.folder+frames[0])).transpose(1, 0, 2))
        
        for frame_id, frame in enumerate(frames[1:]):
            frame = self.convert_frame(np.asarray(Image.open(self.folder+frame)).transpose(1, 0, 2))
            self.frame_to_events(frame_id, frame, reference, threshold_map, events)
            reference = frame
            if (100 * frame_id / len(frames) % 5) == 0:
                print(str(100 * frame_id / len(frames)) + "%...")

        return np.array(events)
            
pix2eve = Pix2Eve("/home/alphat/Desktop/circles/", 10e6 * 1/10)
events = pix2eve.run()

# times = np.diff(events[:, 2])
# imgs = []
# width = np.max(events[:, 0]) + 1
# height = np.max(events[:, 1]) + 1

# time = 0
# img = np.zeros((height, width, 3))
# for i in range(events.shape[0]-1):
#     img[events[i, 1], events[i, 0], events[i, 3]] = 1
#     time += times[i]
#     if time > 1000000:
#         time = 0
#         plt.figure()
#         plt.imshow(img)
#         img = np.zeros((height, width, 3))


