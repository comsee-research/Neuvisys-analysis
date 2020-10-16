#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

from psychopy import visual

def flashing_grating(folder, frequency=1/346, orientation=0, time=1, framerate=1000, flash_period=0.5, flip=False): # time in s
    win = visual.Window([346, 260], screen=0, monitor="testMonitor", fullscr=False, color=[0, 0, 0], units="pix")
    grat_stim = visual.GratingStim(win=win, tex="sqr", units="pix", pos=(0.0, 0.0), size=500)
    grat_stim.sf = frequency
    grat_stim.ori = orientation
    
    phase = 0
    for i in range(int(time*framerate)):
        if i % int(flash_period*framerate) == 0:
            phase = (phase + 0.5) % 1
        grat_stim.phase = phase
        grat_stim.draw()
        win.getMovieFrame(buffer='back')
        if flip:
            win.flip()
    
    win.saveMovieFrames(fileName=folder+"frame.png")
    win.close()

flashing_grating(folder="/home/thomas/Bureau/test/",
                 frequency=70/346, orientation=0, time=1, framerate=1000, flash_period=0.5, flip=False)