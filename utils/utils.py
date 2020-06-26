#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:34:42 2020

@author: thomas
"""

import json
import numpy as np
from dv import AedatFile
from dv import LegacyAedatFile            
import os, shutil

def load_params(param_path):
    with open(param_path) as file:
        return json.load(file)


def load_aedat4(file_path):
    with AedatFile(file_path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    return events


def load_aedat(file_path):
    with LegacyAedatFile(file_path) as f:
        for e in f:
            print(e.x, e.y)


def delete_files(folder):
    for file in os.scandir(folder):
        try:
            if os.path.isfile(file.path) or os.path.islink(file.path):
                os.unlink(file.path)
            elif os.path.isdir(file.path):
                shutil.rmtree(file.path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file.path, e))
    print("deleted files in " + folder)
    
def write_json(file_path, dest):
    events = []
    with AedatFile(file_path) as f:
        for event in f["events"]:
            events.append((event.timestamp, event.x, event.y, event.polarity))

    with open(dest, "w") as file:
        json.dump(events, file)

folder = "/home/thomas/neuvisys-dv/configuration/network/weights/"
delete_files(folder)
folder = "/home/thomas/neuvisys-dv/configuration/network/images/"
delete_files(folder)

# write_json("/home/thomas/Vidéos/pen.aedat4", "/home/thomas/Bureau/test")