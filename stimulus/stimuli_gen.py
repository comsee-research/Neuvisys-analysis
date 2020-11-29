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

def grating_generation(folder, display=False, time=0.2, framerate=1000, flash_period=0.1):
    """
    time # s
    framerate # fps
    flash_period # s
    """
    
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
        if display:
            win.flip()
   
    win.saveMovieFrames(fileName=folder)
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
            
def moving_lines(time=10, framerate=1000, speed=200):
    cnt = 0
    positions = np.linspace(0, 350, 11, dtype=np.uint16)
    
    for frame in range(int(time*framerate)):
        img = np.full((260, 346, 3), 0, np.uint8)
        
        shift = int(frame * (speed / framerate))
        for i in positions:
            pos = (i + shift) % 350
            # cv.line(img, (pos, 0), (pos, 260), (0, 0, 0), 4)
            cv.line(img, (pos+5, 0), (pos+5, 260), (255, 255, 255), 4)
            
        image = Image.fromarray(img)
        image.save("/home/alphat/Desktop/lines/img"+str(cnt)+".png")
        cnt += 1
        
moving_lines()