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