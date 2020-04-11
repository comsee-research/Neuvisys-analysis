#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

from dv import AedatFile
from dv import LegacyAedatFile

import numpy as np

def load_aedat4(file_path):
    with AedatFile(file_path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    return events

def load_aedat(file_path):
    with LegacyAedatFile(file_path) as f:
        for e in f:
            print(e.x, e.y)

import rosbag
from bitarray import bitarray

def write_aedat_header(aedat_file):
    aedat_file.write(b'#!AER-DAT2.0\r\n')
    aedat_file.write(b'# This is a raw AE data file created by saveaerdat.m\r\n')
    aedat_file.write(b'# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
    aedat_file.write(b'# Timestamps tick is 1 us\r\n')
    aedat_file.write(b'# End of ASCII Header\r\n')
    
def format_aedat2(timestamp, x, y, polarity):
    y = format(y, "09b")
    x = format(x, "010b")
    p = "10" if polarity else "00"
    address = bitarray("0" + y + x + p + "0000000000")
    timestamp = bitarray(format(timestamp, "032b"))
    return address, timestamp

def convert_ros_to_aedat(bag_file, aedat_file, x_size, y_size):
    print("\nFormatting: .rosbag -> .aedat\n")
    
    # open the file and write the headers
    with open(aedat_file, "wb") as file:
        write_aedat_header(file)
        
        bag = rosbag.Bag(bag_file)
        
        # setup the camera width and height by adding one event
        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for e in msg.events:
                address, timestamp = format_aedat2(int(e.ts.to_nsec() / 1000.0), x_size-1, y_size-1, 1)
                
                file.write(address.tobytes())
                file.write(timestamp.tobytes())
                break
            break

        # format and write the bag content to the aedat file
        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for e in msg.events:
                address, timestamp = format_aedat2(int(e.ts.to_nsec() / 1000.0), x_size-1-e.x, y_size-1-e.y, e.polarity)
                
                file.write(address.tobytes())
                file.write(timestamp.tobytes())
        bag.close()
        
def concatenate_file(n_concat, aedat4_file, new_file, x_size, y_size):
    events = load_aedat4(aedat4)
    
    first_timestamp = events[0]["timestamp"]
    duration = events[-1]["timestamp"] - first_timestamp
    
    with open(new_file, "wb") as file:
        write_aedat_header(file)
        
        address, timestamp = format_aedat2(first_timestamp, x_size-1, y_size-1, 1)
        file.write(address.tobytes())
        file.write(timestamp.tobytes())
        
        for i in range(n_concat):
            for event in events:
                address, timestamp = format_aedat2(event["timestamp"]+i*duration, x_size-1-event["x"], y_size-1-event["y"], event["polarity"])
                
                file.write(address.tobytes())
                file.write(timestamp.tobytes())
        
bag_file = "/home/thomas/Desktop/Event/files/out.bag"
aedat_file = "/home/thomas/Desktop/Event/files/out.aedat"
# convert_ros_to_aedat(bag_file, aedat_file, 240, 180)

aedat4 = "/home/thomas/neuvisys-analysis/events/files/9_speed.aedat4"
new_file = "/home/thomas/neuvisys-analysis/events/files/9_speed_2.aedat"
aedat = "/home/thomas/Desktop/Event/out.aedat"

event = concatenate_file(10, aedat4, new_file, 240, 180)
