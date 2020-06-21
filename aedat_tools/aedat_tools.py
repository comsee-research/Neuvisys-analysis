#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

import numpy as np
import random

from utils.utils import load_aedat4

import rosbag
from bitarray import bitarray

def write_aedat2_header(aedat_file):
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

def write_aedat2_file(events, outfile, x_size, y_size, new=False):
    print("writing file " + outfile)
    bits = bitarray()
    
    if new:
        first_timestamp = events[0]["timestamp"]
        bits += event_address_aedat2(x_size-1, y_size-1, 1)
        bits += timestamp_aedat2(first_timestamp)
        
        for i in range(events.size):
            event = events[i]
            bits += event_address_aedat2(x_size-1-event["x"], y_size-1-event["y"], event["polarity"])
            bits += timestamp_aedat2(event["timestamp"])
            
        # buffer system
        with open(outfile, "wb") as out:
            write_aedat2_header(out)
            out.write(bits.tobytes())
            bits = bitarray()
    
def convert_ros_to_aedat(bag_file, aedat_file, x_size, y_size, n_concat):
    print("\nFormatting: .rosbag -> .aedat\n")
    
    # open the file and write the headers
    with open(aedat_file, "wb") as file:
        write_aedat2_header(file)
        bag = rosbag.Bag(bag_file)

        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for n, e in enumerate(msg.events):
                if n == 0:
                    file.write(event_address_aedat2(x_size-1, y_size-1, 1).tobytes())
                    file.write(timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())
                    
                file.write(event_address_aedat2(x_size-1-e.x, y_size-1-e.y, e.polarity).tobytes())
                file.write(timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())
        bag.close()
        
def remove_blank_space(aedat4_file, outfile, x_size, y_size):
    events = load_aedat4(aedat4_file)
    times = events["timestamp"]
    
    diff = np.diff(times)
    arg = np.argwhere(diff > 1000000)[0][0]
    times[arg+1:] -= diff[arg]
    
    write_aedat2_file(events, outfile, x_size, y_size)

def concatenate_file(n_concat, aedat4_file, new_file, x_size, y_size):
    events = load_aedat4(aedat4_file)
    
    first_timestamp = events[0]["timestamp"]
    duration = events[-1]["timestamp"] - first_timestamp
    
    with open(new_file, "wb") as file:
        write_aedat2_header(file)
        
        file.write(event_address_aedat2(x_size-1, y_size-1, 1).tobytes())
        file.write(timestamp_aedat2(first_timestamp).tobytes())
        
        for i in range(n_concat):
            for event in events:                
                file.write(event_address_aedat2(x_size-1-event["x"], y_size-1-event["y"], event["polarity"]).tobytes())
                file.write(timestamp_aedat2(event["timestamp"]+i*duration).tobytes())

def divide_events(events, chunk_size):
    first_timestamp = events["timestamp"][0]
    events["timestamp"] -= first_timestamp
    chunk = np.arange(0, events["timestamp"][-1], chunk_size)
    splits = [events[(events["timestamp"] > chunk[i]) & (events["timestamp"] < chunk[i+1])] for i in range(chunk.size-1)]
    
    for split in splits:
        split["timestamp"] -= split["timestamp"][0]
    return splits, first_timestamp

def build_mixed_file(files, chunk_size):
    splits = []
    f_timestamps = []
    
    for file in files:
        div, first_timestamp = divide_events(load_aedat4(file), chunk_size)
        splits += div
        f_timestamps.append(first_timestamp)
    random.shuffle(splits)
    
    for i, split in enumerate(splits):
        split["timestamp"] += i * chunk_size + f_timestamps[0]
        
    return np.hstack(splits)

## Script

files = ["/home/thomas/Videos/driving/city_34.aedat4", "/home/thomas/Videos/driving/freeway_40.aedat4"]
chunk_size = 10000000

events = build_mixed_file(files, chunk_size)
write_aedat2_file(events, "/home/thomas/Desktop/split_test.aedat", 346, 260, True)