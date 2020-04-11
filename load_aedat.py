#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

from dv import AedatFile
from dv import LegacyAedatFile

def load_aedat4(file_path):
    with AedatFile(file_path) as f:
        for e in f['events'].numpy():
            print(e["y"])

def load_aedat(file_path):
    with LegacyAedatFile(file_path) as f:
        for e in f:
            print(e.x, e.y)

aedat4 = "/home/thomas/test.aedat4"
aedat = "/home/thomas/Desktop/Event/out.aedat"

load_aedat(aedat)