#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:55:38 2020

@author: thomas
"""

import PySimpleGUI as sg
import json

def write_json(directory, gui):
    with open(directory, "w") as file:
        json.dump(gui, file)

weight_sharing = True

patch_width = 4
patch_height = 4

if weight_sharing:
    width = 3*patch_width
    height = 3*patch_height
    depth = 5
else:
    width = 34
    height = 26
    depth = 4

width2 = 11
height2 = 8
depth2 = 1

sg.theme('DarkAmber')

layout1 = []

count = 0
if weight_sharing:
    for i in range(height):
        layout1.append([])
    
    for i in range(3):
        for j in range(3):
            for col in range(patch_height):
                for row in range(patch_width):
                    layout1[j*patch_width+row].append(sg.Button("{:>3}".format(str(count)), key="l1"+str(count)))
                    count += 1
            layout1[j*4+0].append(sg.Text(" "))
            layout1[j*4+1].append(sg.Text(" "))
            layout1[j*4+2].append(sg.Text(" "))
            layout1[j*4+3].append(sg.Text(" "))
    layout1.insert(4, [sg.Text(" ")])
    layout1.insert(9, [sg.Text(" ")])
else:
    for i in range(height):
        layout1.append([])
    
    for col in range(width):
        for row in range(height):
            layout1[row].append(sg.Button("{:>3}".format(str(count)), key="l1"+str(count)))
            count += 1
layout1.insert(0, [sg.Slider(range=(0, depth-1), size=(80, 20), orientation="horizontal", key="depth1", enable_events=True)])
layout1.append([sg.Button("Save", key="save")])

count = 0
layout2 = []

for i in range(height2):
    layout2.append([])

for col in range(width2):
    for row in range(height2):
        layout2[row].append(sg.Button("{:>3}".format(str(count)), key="l2"+str(count)))
        count += 1
layout2.insert(0, [sg.Slider(range=(0, depth-1), orientation="horizontal", key="depth2", enable_events=True)])

layout = [[sg.TabGroup([[sg.Tab('Tab 1', layout1), sg.Tab('Tab 2', layout2)]])]]

# Create the Window
win = sg.Window('Neuvisys Interface', layout, default_button_element_size=(1, 1), auto_size_buttons=False)

gui = {"index": 0,
       "index2": 0,
       "layer": 0, 
       "layer2": 0,
       "save": False}

directory = "/home/thomas/neuvisys-dv/configuration/gui.json"
x = 0
y = 0
index = 0
while True:
    event, values = win.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    elif event == "save":
        gui["save"] = True
        write_json(directory, gui)
    elif event == "depth1":
        index = int(x * width * depth + y * depth + values["depth1"])
        gui["index"] = index
        gui["layer"] = int(values["depth1"])
        write_json(directory, gui)
    elif "l1" in event:
        x = int(event[2:]) // width
        y = int(event[2:]) % width
        index = int(x * width * depth + y * depth + values["depth1"])
        gui["index"] = index
        gui["layer"] = int(values["depth1"])
        write_json(directory, gui)
    elif "l2" in event:
        x2 = int(event[2:]) // width2
        y2 = int(event[2:]) % width2
        index2 = int(x2 * width2 * depth2 + y2 * depth2)
        gui["index2"] = index2
        write_json(directory, gui)
win.close()