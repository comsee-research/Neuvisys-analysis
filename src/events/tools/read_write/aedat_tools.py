#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

import numpy as np
from dv import AedatFile
from dv import LegacyAedatFile


def load_aedat(file_path):
    with LegacyAedatFile(file_path) as f:
        for e in f:
            print(e.x, e.y)


def load_aedat4(file_path):
    with AedatFile(file_path) as f:
        events = np.hstack([packet for packet in f["events"].numpy()])
        try:
            events2 = np.hstack([packet for packet in f["events_1"].numpy()])
            return events, events2
        except:
            return events
