#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

from psychopy import visual
import numpy as np
import cv2 as cv
from PIL import Image


def counterphase_grating(win, frequency=1/346, orientation=0, phase=0, contrast=1):
    grat_stim = visual.GratingStim(win=win, tex="sqr", units="pix", pos=(0.0, 0.0), size=500)
    grat_stim.sf = frequency
    grat_stim.ori = orientation
    grat_stim.phase = phase
    grat_stim.contrast = contrast
    grat_stim.draw()

def grating_generation():
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
        
        counterphase_grating(win, 58/346, 0, phase, contrast)
            
        win.getMovieFrame(buffer='back')
        win.flip()
   
    # win.saveMovieFrames(fileName="/home/thomas/Bureau/test/frame.png")
    win.close()

def falling_leaves(nb_circles=1000, framerate=10):
    img = np.zeros((260, 346, 3), np.uint8)
    
    cnt = 0
    for i in range(nb_circles):
        center_x = np.random.randint(0, 346)
        center_y = np.random.randint(0, 260)
        intensity = np.random.randint(0, 255)
        size = np.random.randint(10, 40)
    
        cv.circle(img, (center_x, center_y), size, (intensity, intensity, intensity), 2)
    
        image = Image.fromarray(img)
        for frame in range(framerate):
            image.save("/home/alphat/Desktop/circles/img"+str(cnt)+".png")
            cnt += 1