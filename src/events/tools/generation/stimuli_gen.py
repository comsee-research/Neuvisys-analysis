#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

import cv2 as cv
import numpy as np
from PIL import Image
#from psychopy import visual
from scipy import ndimage


def counterphase_grating(win, frequency=1 / 346, orientation=0, phase=0, contrast=1):
    grat_stim = visual.GratingStim(
        win=win, tex="sqr", units="pix", pos=(0.0, 0.0), size=500
    )
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

    x = np.sin(np.linspace(-np.pi / 2, np.pi / 2, int(flash_period * framerate) // 2))
    flash = (np.hstack((x, x[::-1])) + 1) / 2
    phases = [0, 0.5]

    win = visual.Window(
        [346, 260],
        screen=0,
        monitor="testMonitor",
        fullscr=False,
        color=[0, 0, 0],
        units="pix",
    )
    switch = 0
    for i in range(int(time * framerate)):
        if i % int(flash_period * framerate) == 0:
            switch = not switch
        index = i % int(flash_period * framerate)
        contrast = flash[index]
        phase = phases[switch]

        counterphase_grating(win, 58 / 346, 0, phase, contrast)

        win.getMovieFrame(buffer="back")
        if display:
            win.flip()

    win.saveMovieFrames(fileName=folder)
    win.close()


def falling_leaves(time=10, framerate=1000, nb_circle_frame=4):
    img = np.full((260, 346, 3), 127, np.uint8)

    cnt = 0
    for frame in range(int(time * framerate)):
        for i in range(nb_circle_frame):
            center_x = np.random.randint(0, 346)
            center_y = np.random.randint(0, 260)
            intensity = np.random.randint(0, 255)
            size = np.random.randint(10, 40)
            cv.circle(
                img, (center_x, center_y), size, (intensity, intensity, intensity), 2
            )

        image = Image.fromarray(img)
        image.save("/home/alphat/Desktop/circles/img" + str(cnt) + ".png")
        cnt += 1


def moving_lines(folder, time=10, framerate=1000, speed=200, rotation=0, disparity=0, frame_start=0):
    cnt = frame_start
    positions = np.linspace(0, 550, 15, dtype=np.uint16)

    for frame in range(int(time * framerate)):
        img = np.full((460, 550, 3), 0, np.uint8)

        shift = int(frame * (speed / framerate))
        for i in positions:
            pos = (i + shift) % 550
            cv.line(img, (pos + disparity, 0), (pos + disparity, 460), (255, 255, 255), 4)

        img = ndimage.rotate(img, rotation, reshape=False, order=0)

        image = Image.fromarray(img[100:360, 100:446])
        image.save(folder + "img" + str(cnt) + ".png")
        cnt += 1


def moving_bars(folder, framerate=1000, speeds=None):
    if speeds is None:
        speeds = [400, 200, 100, 50]
    frame = 0
    y = np.linspace(0, 260, len(speeds) + 1, dtype=np.uint16)
    shift = 0

    while shift < 350:
        img = np.full((260, 346, 3), 0, np.uint8)

        for i, speed in enumerate(speeds):
            shift = int(frame * (speed / framerate))
            cv.line(img, (0 + shift, y[i]), (0 + shift, y[i + 1]), (255, 255, 255), 4)

        # img = ndimage.rotate(img, rotation, reshape=False, order=0)

        image = Image.fromarray(img)
        image.save(folder + "img" + str(frame) + ".png")
        frame += 1


def disparity_bars(folder, framerate=1000, speeds=None, disparities=None):
    if disparities is None:
        disparities = [8, 6, 4, 2]
    if speeds is None:
        speeds = [400, 200, 100, 50]
    frame = 0
    x = np.array(disparities, dtype=np.uint16)
    y = np.linspace(0, 260, len(disparities) + 1, dtype=np.uint16)
    shift = 0

    while shift < 350:
        img = np.full((260, 346, 3), 0, np.uint8)

        for i, speed in enumerate(speeds):
            shift = int(frame * (speed / framerate))
            cv.line(img, (x[i] + shift, y[i]), (x[i] + shift, y[i + 1]), (255, 255, 255), 4)

        # img = ndimage.rotate(img, rotation, reshape=False, order=0)

        image = Image.fromarray(img)
        image.save(folder + "img" + str(frame) + ".png")
        frame += 1
