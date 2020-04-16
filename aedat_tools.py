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
    
def format_address_aedat2(x, y, polarity):
    y = format(y, "09b")
    x = format(x, "010b")
    p = "10" if polarity else "00"
    return bitarray("0" + y + x + p + "0000000000")

def format_timestamp_aedat2(timestamp):
    return bitarray(format(timestamp, "032b"))

def convert_ros_to_aedat(bag_file, aedat_file, x_size, y_size, n_concat):
    print("\nFormatting: .rosbag -> .aedat\n")
    
    # open the file and write the headers
    with open(aedat_file, "wb") as file:
        write_aedat_header(file)
        bag = rosbag.Bag(bag_file)

        # format and write the bag content to the aedat file
        addresses = []
        timestamps = []
        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for n, e in enumerate(msg.events):
                if n == 0:
                    file.write(format_address_aedat2(x_size-1, y_size-1, 1).tobytes())
                    file.write(format_timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())
                    
                file.write(format_address_aedat2(x_size-1-e.x, y_size-1-e.y, e.polarity).tobytes())
                file.write(format_timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())
                # addresses.append(format_address_aedat2(x_size-1-e.x, y_size-1-e.y, e.polarity))
                # timestamps.append(int(e.ts.to_nsec() / 1000.0))
        bag.close()
        
        # duration = timestamps[-1] - timestamps[0]
        # print(duration)
        
        # for i in range(n_concat):
        #     for address, timestamp in zip(addresses, timestamps):
        #         print(address)
        #         print(timestamp)
        #         file.write(address.tobytes())
        #         file.write(format_timestamp_aedat2(timestamp+i*duration).tobytes())

def concatenate_file(n_concat, aedat4_file, new_file, x_size, y_size):
    events = load_aedat4(aedat4_file)
    
    first_timestamp = events[0]["timestamp"]
    duration = events[-1]["timestamp"] - first_timestamp
    
    with open(new_file, "wb") as file:
        write_aedat_header(file)
        
        file.write(format_address_aedat2(x_size-1, y_size-1, 1).tobytes())
        file.write(format_timestamp_aedat2(first_timestamp).tobytes())
        
        for i in range(n_concat):
            for event in events:
                file.write(format_address_aedat2(x_size-1-event["x"], y_size-1-event["y"], event["polarity"]).tobytes())
                file.write(format_timestamp_aedat2(event["timestamp"]+i*duration).tobytes())
        
bag_file = "/home/thomas/neuvisys-analysis/events/files/out.bag"
aedat_file = "/home/thomas/neuvisys-analysis/events/files/out.aedat"
# convert_ros_to_aedat(bag_file, aedat_file, 346, 260, 1)

aedat_4 = "/home/thomas/neuvisys-analysis/events/files/temp.aedat4"
new_file = "/home/thomas/neuvisys-analysis/events/files/out.aedat"
concatenate_file(20, aedat_4, new_file, 346, 260)
