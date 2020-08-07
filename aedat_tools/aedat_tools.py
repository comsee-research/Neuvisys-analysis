#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

from dv import AedatFile
from dv import LegacyAedatFile
import json
import os, shutil
import numpy as np
import random
import rosbag
from bitarray import bitarray

def delete_files(path):
    for file in os.scandir(path):
        try:
            if os.path.isfile(file.path) or os.path.islink(file.path):
                os.unlink(file.path)
            elif os.path.isdir(file.path):
                shutil.rmtree(file.path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file.path, e))

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

def write_npdat(events, dest):    
    arr = np.zeros((events.size, 4))
    arr[:, 0] = events["timestamp"]
    arr[:, 1] = events["x"]
    arr[:, 2] = events["y"]
    arr[:, 3] = events["polarity"]
    
    with open(dest, "wb") as file:
        np.save(file, arr)

def write_aedat2_header(aedat_file):
    aedat_file.write(b'#!AER-DAT2.0\r\n')
    aedat_file.write(b'# This is a raw AE data file created by saveaerdat.m\r\n')
    aedat_file.write(b'# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
    aedat_file.write(b'# Timestamps tick is 1 us\r\n')
    aedat_file.write(b'# End of ASCII Header\r\n')
    
def event_address_aedat2(x, y, polarity):
    y = format(y, "09b")
    x = format(x, "010b")# write_npdat("/home/thomas/Vidéos/driving_dataset/mix/mix_17.aedat4", "/home/thomas/Vidéos/driving_dataset/npy/mix_17.npy")
    p = "10" if polarity else "00"
    return bitarray("0" + y + x + p + "0000000000")

def timestamp_aedat2(timestamp):
    return bitarray(format(timestamp, "032b"))

def frame_address_aedat2(x, y, intensity):
    y = format(y, "09b")
    x = format(x, "010b")
    intensity = format(intensity, "010b")
    return bitarray("1" + y + x + "10" + intensity)

def write_aedat2_file(events, outfile, x_size, y_size):
    print("writing file " + outfile)
    bits = bitarray()
    
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
    
def convert_ros_to_aedat(bag_file, aedat_file, x_size, y_size):
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

def concatenate_files(aedat4_files, outfile, x_size, y_size):
    list_events = []
    duration = 0
    
    for i, file in enumerate(aedat4_files):
        events = load_aedat4(file)
        events["timestamp"] += duration
        duration += events[-1]["timestamp"] - events[0]["timestamp"]
        list_events.append(events)
    events = np.hstack(list_events)
    
    write_aedat2_file(events, outfile, x_size, y_size)

def divide_events(events, chunk_size):
    first_timestamp = events["timestamp"][0]
    events["timestamp"] -= first_timestamp
    chunk = np.arange(0, events["timestamp"][-1], chunk_size)
    splits = [events[(events["timestamp"] > chunk[i]) & (events["timestamp"] < chunk[i+1])] for i in range(chunk.size-1)]
    
    for split in splits:
        try:
            split["timestamp"] -= split["timestamp"][0]
        except:
            print("oups")
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