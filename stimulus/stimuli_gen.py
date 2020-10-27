#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

from psychopy import visual
import numpy as np


def flashing_grating(win, frequency=1/346, orientation=0, phase=0, contrast=1):
    grat_stim = visual.GratingStim(win=win, tex="sqr", units="pix", pos=(0.0, 0.0), size=500)
    grat_stim.sf = frequency
    grat_stim.ori = orientation
    grat_stim.phase = phase
    grat_stim.contrast = contrast
    grat_stim.draw()


time = 1 # s
framerate = 1000 # fps
flash_period = 0.03 # s

x = np.sin(np.linspace(-np.pi/2, np.pi/2, int(flash_period*framerate) // 2))
flash = (np.hstack((x, x[::-1])) + 1) / 2
phases = [0, 0.5]

win = visual.Window([346, 260], screen=0, monitor="testMonitor", fullscr=False, color=[0, 0, 0], units="pix")
switch = 0
for i in range(int(time*framerate)):
    if i % int(flash_period*framerate) == 0:
        switch = not switch
    index = i % int(flash_period*framerate)
    contrast = flash[index]
    phase = phases[switch]
    
    flashing_grating(win, 58/346, 0, phase, contrast)
        
    win.getMovieFrame(buffer='back')
    win.flip()


# win.saveMovieFrames(fileName="/home/thomas/Bureau/test/frame.png")
win.close()

