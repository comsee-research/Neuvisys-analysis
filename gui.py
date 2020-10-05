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

def launch_gui(spinet):  
    weight_sharing = spinet.net_var["WeightSharing"]
    width = spinet.net_var["L1Width"]
    height = spinet.net_var["L1Height"]
    depth = spinet.net_var["L1Depth"]
    
    width2 = spinet.net_var["L2Width"]
    height2 = spinet.net_var["L2Height"]
    depth2 = 1
    
    sg.theme('DarkAmber')
    
    layout1 = []
    
    count = 0
    if weight_sharing:
        for i in range(len(spinet.net_var["L1YAnchor"])*height):
            layout1.append([])
        
        for i in range(len(spinet.net_var["L1XAnchor"])):
            for j in range(len(spinet.net_var["L1YAnchor"])):
                for col in range(height):
                    for row in range(width):
                        layout1[j*width+row].append(sg.Button("{:>3}".format(str(count)), key="l1"+str(count)))
                        count += 1
                for k in range(width):
                    layout1[j*width+k].append(sg.Text(" "))

        for i in range(1, len(spinet.net_var["L1YAnchor"])):
            layout1.insert(i*height+i-1, [sg.Text(" ")])
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