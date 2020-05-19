#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

import numpy as np

from bitarray import bitarray

def load_aedat4(file_path):
    with AedatFile(file_path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    return events

def write_aedat_header(aedat_file):
    aedat_file.write(b'#!AER-DAT2.0\r\n')
    aedat_file.write(b'# This is a raw AE data file created by saveaerdat.m\r\n')
    aedat_file.write(b'# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
    aedat_file.write(b'# Timestamps tick is 1 us\r\n')
    aedat_file.write(b'# End of ASCII Header\r\n')
    
def event_address_aedat2(x, y, polarity):
    y = format(y, "09b")
    x = format(x, "010b")
    p = "10" if polarity else "00"
    return bitarray("0" + y + x + p + "0000000000")

def timestamp_aedat2(timestamp):
    return bitarray(format(timestamp, "032b"))

def frame_address_aedat2(x, y, intensity):
    y = format(y, "09b")
    x = format(x, "010b")
    intensity = format(intensity, "010b")
    return bitarray("1" + y + x + "10" + intensity)
