#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

from psychopy import visual, core, event

win0 = visual.Window([346, 260], screen=0, monitor="testMonitor", fullscr=False, color=[0, 0, 0], units="pix")

grat_stim_h = visual.GratingStim(win=win0, tex="sqr", units="pix", pos=(0.0, 0.0), size=346, sf=0.09, ori=0.0, phase=(0.0, 0.0))
grat_stim_v = visual.GratingStim(win=win0, tex="sqr", units="pix", pos=(0.0, 0.0), size=346, sf=0.09, ori=90.0, phase=(0.0, 0.0))

i = 0
switch = True

while(i < 10000):
    if i % 1000 == 0:
        switch = not switch
    
    if switch:
        grat_stim_h.setPhase(0.002, "+")
        grat_stim_h.draw()
    else:
        grat_stim_v.setPhase(0.002, "+")
        grat_stim_v.draw()
    
    win0.getMovieFrame(buffer='back')
    win0.flip()
    
    if len(event.getKeys()) > 0:
        break
    event.clearEvents()
    
    i += 1
    
win0.saveMovieFrames(fileName='/home/thomas/Bureau/frames/frame.png')

win0.close()
