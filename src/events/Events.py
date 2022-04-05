#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:23:51 2022

@author: thomas
"""

from pathlib import Path

import h5py
import numpy as np
import skvideo.io
from dv import AedatFile
from tqdm import tqdm


def npz_to_ndarray(npz):
    left_events = np.zeros((npz["arr_0"].shape[0], 4))
    if len(npz.keys()) > 4:
        right_events = np.zeros((npz["arr_4"].shape[0], 4))
    else:
        right_events = np.array([])
    for i, k in enumerate(npz.keys()):
        if i < 4:
            left_events[:, i] = npz[k]
        else:
            right_events[:, i - 4] = npz[k]
    if right_events.any():
        return left_events, right_events
    else:
        return left_events


def npaedat_to_ndarray(events):
    eve = np.zeros((events["timestamp"].shape[0], 4))
    eve[:, 0] = events["timestamp"]
    eve[:, 1] = events["x"]
    eve[:, 2] = events["y"]
    eve[:, 3] = events["polarity"]
    return eve


def load_aedat4(filepath):
    with AedatFile(filepath) as f:
        events = np.hstack([packet for packet in f["events"].numpy()])
        try:
            events2 = np.hstack([packet for packet in f["events_1"].numpy()])
            return events, events2
        except:
            return events


def load_h5(filepath):
    h5f = h5py.File(str(filepath), 'r')
    events = dict()
    for dset_str in ['p', 'x', 'y', 't']:
        events[dset_str] = h5f['events/{}'.format(dset_str)]
    return events


def write_npz(dest, events):
    if type(events) is tuple:
        try:
            np.savez(
                dest,
                events[0]["timestamp"].astype("i8"),
                events[0]["x"].astype("i2"),
                events[0]["y"].astype("i2"),
                events[0]["polarity"].astype("i1"),
                events[1]["timestamp"].astype("i8"),
                events[1]["x"].astype("i2"),
                events[1]["y"].astype("i2"),
                events[1]["polarity"].astype("i1"),
            )
        except (IndexError, KeyError) as e:
            try:
                np.savez(
                    dest,
                    events[0]["arr_0"].astype("i8"),
                    events[0]["arr_1"].astype("i2"),
                    events[0]["arr_2"].astype("i2"),
                    events[0]["arr_3"].astype("i1"),
                    events[1]["arr_0"].astype("i8"),
                    events[1]["arr_1"].astype("i2"),
                    events[1]["arr_2"].astype("i2"),
                    events[1]["arr_3"].astype("i1"),
                )
            except (IndexError, KeyError) as e:
                try:
                    np.savez(
                        dest,
                        events[0][:, 0].astype("i8"),
                        events[0][:, 1].astype("i2"),
                        events[0][:, 2].astype("i2"),
                        events[0][:, 3].astype("i1"),
                        events[1][:, 0].astype("i8"),
                        events[1][:, 1].astype("i2"),
                        events[1][:, 2].astype("i2"),
                        events[1][:, 3].astype("i1"),
                    )
                except (IndexError, KeyError) as e:
                    raise
    else:
        try:
            np.savez(
                dest,
                events["timestamp"].astype("i8"),
                events["x"].astype("i2"),
                events["y"].astype("i2"),
                events["polarity"].astype("i1"),
            )
        except (IndexError, KeyError) as e:
            try:
                np.savez(
                    dest,
                    events[0]["arr_0"].astype("i8"),
                    events[0]["arr_1"].astype("i2"),
                    events[0]["arr_2"].astype("i2"),
                    events[0]["arr_3"].astype("i1"),
                )
            except (IndexError, KeyError) as e:
                try:
                    np.savez(
                        dest,
                        events[:, 0].astype("i8"),
                        events[:, 1].astype("i2"),
                        events[:, 2].astype("i2"),
                        events[:, 3].astype("i1"),
                    )
                except (IndexError, KeyError) as e:
                    raise


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, height: int, width: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert height > 0
    assert width > 0
    img = np.full((height, width, 3), fill_value=255, dtype='uint8')
    mask = np.zeros((height, width), dtype='int32')
    pol = pol.astype('int')
    x = x.astype('int')
    y = y.astype('int')
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (width > x) & (height > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [255, 255, 255]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0, 255]
    return img


class Events:
    def __init__(self, event_files, cameras):
        self.dtype = np.dtype([('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1'), ('c', 'u1')])
        self.event_array = np.zeros(0, self.dtype)
        
        for event_file, camera in zip(event_files, cameras):
            self.add_events(event_file, camera)

    def load_hdf5(self, filepath, camera):
        file = h5py.File(str(filepath), 'r')
        event_dataset = file["events"]
        
        event_array = np.zeros(event_dataset["t"].size, self.dtype)
        event_array['t'] = np.asarray(event_dataset["t"])
        event_array['x'] = np.asarray(event_dataset["x"])
        event_array['y'] = np.asarray(event_dataset["y"])
        event_array['p'] = np.asarray(event_dataset["p"])
        event_array['c'].fill(camera)
        
        self.event_array = np.hstack((self.event_array, event_array))
        
    def add_events(self, event_file, camera):
        if isinstance(event_file, np.ndarray):
            self.event_array = event_file
        if isinstance(event_file, str):
            if event_file.endswith(".npz"):
                npz_to_ndarray(np.load(event_file))
            elif event_file.endswith(".h5"):
                self.load_hdf5(event_file, camera)
                
    def sort_events(self):
        self.event_array = self.event_array[self.event_array["t"].argsort()]
        
    def get_events(self):
        return self.event_array
    
    def get_timestamps(self):
        return self.event_array['t']
    
    def get_x(self):
        return self.event_array['x']
    
    def get_y(self):
        return self.event_array['y']
    
    def get_polarities(self):
        return self.event_array['p']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self.event_array.__iter__()

    def __next__(self):
        return self.event_array.__next__()

    def save_as_file(self, dest, extension):
        if extension == ".npz":
            np.savez(
                dest,
                self.events["t"].astype("i8"),
                self.events["x"].astype("i2"),
                self.events["y"].astype("i2"),
                self.events["p"].astype("i1"),
                self.events["c"].astype("i1"),
            )
        elif extension == ".h5":
            with h5py.File(dest + ".h5", "a") as file :
                group = file.create_group("events")
                group.create_dataset("t", self.event_array["t"].shape, dtype=self.event_array["t"].dtype, data=self.event_array["t"], compression="gzip")
                group.create_dataset("x", self.event_array["x"].shape, dtype=self.event_array["x"].dtype, data=self.event_array["x"], compression="gzip")
                group.create_dataset("y", self.event_array["y"].shape, dtype=self.event_array["y"].dtype, data=self.event_array["y"], compression="gzip")
                group.create_dataset("p", self.event_array["p"].shape, dtype=self.event_array["p"].dtype, data=self.event_array["p"], compression="gzip")
                group.create_dataset("c", self.event_array["c"].shape, dtype=self.event_array["c"].dtype, data=self.event_array["c"], compression="gzip")

    def to_video(self, dt_miliseconds, dest, width, height):
        writer = skvideo.io.FFmpegWriter(Path(dest + ".mp4"))
        for events in tqdm(EventSlicer(self.get_events(), dt_miliseconds)):
            img = render(events["x"], events["y"], events["p"], height, width)
            writer.writeFrame(img)
        writer.close()

    def remove_events(self, timestamp_starts, timestamp_end):
        for i, j in zip(timestamp_starts, timestamp_end):
            self.event_array = np.delete(
                self.event_array, (self.event_array["t"] >= i) & (self.event_array["t"] <= j)
            )

    def resize_events(self, w_start, h_start, width, height):
        self.event_array = np.delete(self.event_array,
                                (self.event_array["x"] < w_start) | (self.event_array["x"] >= w_start + width) |
                                (self.event_array["y"] < h_start) | (self.event_array["y"] >= h_start + height),
                                axis=0)
        self.event_array["x"] -= np.min(self.event_array["x"])
        self.event_array["y"] -= np.min(self.event_array["y"])

    def _finalizer(self):
        pass


class EventSlicer:
    def __init__(self, events, dt_milliseconds: int):
        self.events = events
        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.events["t"][0]
        self.t_end_us = self.events["t"][-1]

        self._length = (self.t_end_us - self.t_start_us) // self.dt_us

    def __len__(self):
        return self._length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def __next__(self):
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration

        self.t_start_us = t_end_us
        return events

    def get_events(self, t_start_us: int, t_end_us: int):
        return self.events[(self.events["t"] > t_start_us) & (self.events["t"] < t_end_us)]

    def _finalizer(self):
        pass
