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
# import rosbag
from bitarray import bitarray
from PIL import Image, ImageDraw
import cv2 as cv

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
        try:
            events2 = np.hstack([packet for packet in f['events_1'].numpy()])
            return events, events2
        except:
            return events
        
def write_npz(dest, events):
    if type(events) is tuple:
        try:
            np.savez(dest, events[0]["timestamp"].astype('i8'), events[0]["x"].astype('i2'), events[0]["y"].astype('i2'), events[0]["polarity"].astype('i1'),
                     events[1]["timestamp"].astype('i8'), events[1]["x"].astype('i2'), events[1]["y"].astype('i2'), events[1]["polarity"].astype('i1'))
        except:
            try:
                np.savez(dest, events[0][:, 0].astype('i8'), events[0][:, 1].astype('i2'), events[0][:, 2].astype('i2'), events[0][:, 3].astype('i1'),
                         events[1][:, 0].astype('i8'), events[1][:, 1].astype('i2'), events[1][:, 2].astype('i2'), events[1][:, 3].astype('i1'))
            except:
                raise
    else:        
        try:
            np.savez(dest, events["timestamp"].astype('i8'), events["x"].astype('i2'), events["y"].astype('i2'), events["polarity"].astype('i1'))
        except:
            try:
                np.savez(dest, events[:, 0].astype('i8'), events[:, 1].astype('i2'), events[:, 2].astype('i2'), events[:, 3].astype('i1'))
            except:
                raise
        
def load_aedat(file_path):
    with LegacyAedatFile(file_path) as f:
        for e in f:
            print(e.x, e.y)
            
def txt_to_events(file_path):
    arr = []
    with open(file_path, "r") as file:
        for line in file:
            event = line.split(" ")
            arr.append([int(event[0]), int(event[1]), int(event[2]), int(event[3][0])])
    events = np.array(arr)
    events = events[events[:, 2].argsort()]
    return events

def h5py_to_npy(events):
    npy_events = np.zeros(events.shape[0], dtype=([("timestamp", "i8"), ("x", "i8"), ("y", "i8"), ("polarity", "i1")]))
    npy_events["timestamp"] = np.array(1e7 * events[:, 2], dtype="i8")
    npy_events["x"] = np.array(events[:, 0], dtype="i8")
    npy_events["y"] = np.array(events[:, 1], dtype="i8")
    npy_events["polarity"] = np.array((events[:, 3] + 1) / 2, dtype="i1")
    
    return npy_events

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

def write_aedat2_file(events, outfile, x_size, y_size):
    print("writing file " + outfile)
    bits = bitarray()
    
    bits += event_address_aedat2(x_size-1, y_size-1, 1)
    bits += timestamp_aedat2(events[0]["timestamp"])
    
    for event in events:
        bits += event_address_aedat2(x_size-1-event["x"], y_size-1-event["y"], event["polarity"])
        bits += timestamp_aedat2(event["timestamp"])
        
    with open(outfile, "wb") as out:
        write_aedat2_header(out)
        out.write(bits.tobytes())

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

def concatenate_files(aedat4_files):
    list_events = []
    last_tmsp = 0
   
    for i, file in enumerate(aedat4_files):
        events = load_aedat4(file)
        if i != 0:
            events["timestamp"] += last_tmsp - events[0]["timestamp"]
        last_tmsp = events[-1]["timestamp"]
        list_events.append(events)
    return np.hstack(list_events)

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

def npaedat_to_np(events):
    eve = np.zeros((events["timestamp"].shape[0], 4))
    eve[:, 0] = events["timestamp"]
    eve[:, 1] = events["x"]
    eve[:, 2] = events["y"]
    eve[:, 3] = events["polarity"]
    return eve

def npz_to_arr(t, x, y, p):
    eve = np.zeros((t.shape[0], 4))
    eve[:, 0] = t
    eve[:, 1] = x
    eve[:, 2] = y
    eve[:, 3] = p
    return eve
    
def show_event_images(events, time_gap, width, height, dest, rec, side):    
    cnt = 0
    time = 0
    img = np.zeros((height, width, 3))
    
    for time in range(int(events[0, 0]), int(events[-1, 0]), time_gap):
        for event in events[(events[:, 0] >= time) & (events[:, 0] < time + time_gap)]:
            img[int(event[2]), int(event[1]), int(event[3])] = 1
        
        img = img.astype(np.uint8) * 255
        pilim = Image.fromarray(img)
        draw = ImageDraw.Draw(pilim)
        for x in rec[0]:
            for y in rec[1]:
                draw.rectangle([x, y, x + 31, y + 31], outline=(255, 255, 255, 0))
        pilim.save(dest+"img"+str(cnt)+side+"_"+str(time)+".png")
        cnt += 1
        time += time_gap
        img = np.zeros((height, width, 3))
        
def load_frames(path):
    with AedatFile(path) as f:
        try:
            lframes = []
            rframes = []
            for l, r in zip(f["frames"], f["frames_1"]):
                lframes.append(l.image)
                rframes.append(r.image)
            return np.array(lframes), np.array(rframes)
        except:
            frames = []
            for fr in f["frames"]:
                frames.append(fr.image)
            return np.array(frames)
        
def rectify_events(events, lx, ly, rx, ry):
    events[0]["x"] += lx
    events[0]["y"] += ly
    events[1]["x"] += rx
    events[1]["y"] += ry
    
    l_events = events[0][(events[0]["x"] < 346) & (events[0]["x"] >= 0) & (events[0]["y"] < 260) & (events[0]["y"] >= 0)]
    r_events = events[1][(events[1]["x"] < 346) & (events[1]["x"] >= 0) & (events[1]["y"] < 260) & (events[1]["y"] >= 0)]
    
    return l_events, r_events

def rectify_frames(frames, lx, ly, rx, ry):
    rect_frames = np.array(frames).copy()
    rect_frames[0] = shift(rect_frames[0], ly, 1)
    rect_frames[0] = shift(rect_frames[0], lx, 2)
    rect_frames[1] = shift(rect_frames[1], ry, 1)
    rect_frames[1] = shift(rect_frames[1], rx, 2)
    
    return rect_frames

def shift(arr, num, axis, fill_value=0):
    arr = np.roll(arr, num, axis=axis)
    if num < 0:
        if axis == 1:
            arr[:, num:] = fill_value
        if axis == 2:
            arr[:, :, num:] = fill_value
    elif num > 0:
        if axis == 1:
            arr[:, :num] = fill_value
        if axis == 2:
            arr[:, :, :num] = fill_value
    return arr

def remove_events(events, tss, tse):    
    l_events = events[0]
    r_events = events[1]
    
    for i, j in zip(tss, tse):
        l_events = np.delete(l_events, (l_events["timestamp"] >= i) & (l_events["timestamp"] <= j))
        r_events = np.delete(r_events, (r_events["timestamp"] >= i) & (r_events["timestamp"] <= j))
        
    return l_events, r_events

def write_frames(dest, frames):
    for i in range(frames.shape[1]):
        cv.imwrite(dest+"img"+str(i)+"_left.jpg", frames[0][i])
        cv.imwrite(dest+"img"+str(i)+"_right.jpg", frames[1][i])