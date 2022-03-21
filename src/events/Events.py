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


def h5py_to_ndarray(events):
    npy_events = np.zeros((events["p"].shape[0], 4))
    npy_events[:, 0] = np.asarray(events["t"])
    npy_events[:, 1] = np.asarray(events["x"])
    npy_events[:, 2] = np.asarray(events["y"])
    npy_events[:, 3] = np.asarray(events["p"])
    return npy_events


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
    def __init__(self, l_events, r_events=None):
        if r_events is not None:
            self.stereo = True
            if isinstance(l_events, np.ndarray):
                self.l_events = l_events
            if isinstance(r_events, np.ndarray):
                self.r_events = r_events
            if isinstance(l_events, str):
                if l_events.endswith(".npz"):
                    self.l_events = npz_to_ndarray(np.load(l_events))
                elif l_events.endswith(".h5"):
                    self.l_events = h5py_to_ndarray(load_h5(l_events))
            if isinstance(r_events, str):
                if r_events.endswith(".npz"):
                    self.r_events = npz_to_ndarray(np.load(r_events))
                elif r_events.endswith(".h5"):
                    self.r_events = h5py_to_ndarray(load_h5(r_events))
        else:
            self.stereo = False
            if isinstance(l_events, np.ndarray):
                self.events = l_events
            elif isinstance(l_events, str):
                if l_events.endswith(".npz"):
                    eve = np.load(l_events)
                    if len(eve.keys()) > 4:
                        self.stereo = True
                        self.l_events, self.r_events = npz_to_ndarray(eve)
                    else:
                        self.events = npz_to_ndarray(eve)
                elif l_events.endswith(".h5"):
                    self.events = h5py_to_ndarray(load_h5(l_events))
                elif l_events.endswith(".aedat4"):
                    self.events = npaedat_to_ndarray(load_aedat4(l_events))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self.events.__iter__()

    def __next__(self):
        return self.events.__next__()

    def save_file(self, dest):
        if self.stereo:
            np.savez(
                dest,
                self.l_events[:, 0].astype("i8"),
                self.l_events[:, 1].astype("i2"),
                self.l_events[:, 2].astype("i2"),
                self.l_events[:, 3].astype("i1"),
                self.r_events[:, 0].astype("i8"),
                self.r_events[:, 1].astype("i2"),
                self.r_events[:, 2].astype("i2"),
                self.r_events[:, 3].astype("i1"),
            )
        else:
            np.savez(
                dest,
                self.events[:, 0].astype("i8"),
                self.events[:, 1].astype("i2"),
                self.events[:, 2].astype("i2"),
                self.events[:, 3].astype("i1"),
            )

    def to_video(self, dt_miliseconds, dest, width, height):
        if self.stereo:
            writer = skvideo.io.FFmpegWriter(Path(dest + "_left.mp4"))
            for events in tqdm(EventSlicer(self.l_events, dt_miliseconds)):
                img = render(events[:, 1], events[:, 2], events[:, 3], height, width)
                writer.writeFrame(img)
            writer.close()

            writer = skvideo.io.FFmpegWriter(Path(dest + "_right.mp4"))
            for events in tqdm(EventSlicer(self.r_events, dt_miliseconds)):
                img = render(events[:, 1], events[:, 2], events[:, 3], height, width)
                writer.writeFrame(img)
            writer.close()
        else:
            writer = skvideo.io.FFmpegWriter(Path(dest + ".mp4"))
            for events in tqdm(EventSlicer(self.events, dt_miliseconds)):
                img = render(events[:, 1], events[:, 2], events[:, 3], height, width)
                writer.writeFrame(img)
            writer.close()

    def remove_events(self, timestamp_starts, timestamp_end):
        for i, j in zip(timestamp_starts, timestamp_end):
            if self.stereo:
                self.l_events = np.delete(
                    self.l_events, (self.l_events[:, 0] >= i) & (self.l_events[:, 0] <= j)
                )
                self.r_events = np.delete(
                    self.r_events, (self.r_events[:, 0] >= i) & (self.r_events[:, 0] <= j)
                )
            else:
                self.events = np.delete(
                    self.events, (self.events[:, 0] >= i) & (self.events[:, 0] <= j)
                )

    def resize_events(self, w_start, h_start, width, height):
        if self.stereo:
            self.l_events = np.delete(self.l_events,
                                      (self.l_events[:, 1] < w_start) | (self.l_events[:, 1] >= w_start + width) |
                                      (self.l_events[:, 2] < h_start) | (self.l_events[:, 2] >= h_start + height),
                                      axis=0)
            self.r_events = np.delete(self.r_events,
                                      (self.r_events[:, 1] < w_start) | (self.r_events[:, 1] >= w_start + width) |
                                      (self.r_events[:, 2] < h_start) | (self.r_events[:, 2] >= h_start + height),
                                      axis=0)
            self.l_events[:, 1] -= np.min(self.l_events[:, 1])
            self.l_events[:, 2] -= np.min(self.l_events[:, 2])
            self.r_events[:, 1] -= np.min(self.r_events[:, 1])
            self.r_events[:, 2] -= np.min(self.r_events[:, 2])
        else:
            self.events = np.delete(self.events,
                                    (self.events[:, 1] < w_start) | (self.events[:, 1] >= w_start + width) |
                                    (self.events[:, 2] < h_start) | (self.events[:, 2] >= h_start + height),
                                    axis=0)
            self.events[:, 1] -= np.min(self.events[:, 1])
            self.events[:, 2] -= np.min(self.events[:, 2])

    def _finalizer(self):
        pass


class EventSlicer:
    def __init__(self, events, dt_milliseconds: int):
        self.events = events
        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.events[0, 0]
        self.t_end_us = self.events[-1, 0]

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
        return self.events[(self.events[:, 0] > t_start_us) & (self.events[:, 0] < t_end_us)]

    def _finalizer(self):
        pass
