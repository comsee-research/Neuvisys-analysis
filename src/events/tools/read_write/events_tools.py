import json
import os
import shutil

import numpy as np
import rosbag


def delete_files(path):
    for file in os.scandir(path):
        try:
            if os.path.isfile(file.path) or os.path.islink(file.path):
                os.unlink(file.path)
            elif os.path.isdir(file.path):
                shutil.rmtree(file.path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file.path, e))


def load_params(param_path):
    with open(param_path) as file:
        return json.load(file)


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
    npy_events = np.zeros(
        events.shape[0],
        dtype=([("timestamp", "i8"), ("x", "i8"), ("y", "i8"), ("polarity", "i1")]),
    )
    npy_events["timestamp"] = np.array(1e7 * events[:, 2], dtype="i8")
    npy_events["x"] = np.array(events[:, 0], dtype="i8")
    npy_events["y"] = np.array(events[:, 1], dtype="i8")
    npy_events["polarity"] = np.array((events[:, 3] + 1) / 2, dtype="i1")

    return npy_events


def ros_to_npy(bag_file, topic):
    bag = rosbag.Bag(bag_file)
    npy_events = []

    for topic, msg, t in bag.read_messages(topics=[topic]):
        for event in msg.events:
            npy_events.append([event.ts.to_nsec(), event.x, event.y, event.polarity])
    return np.array(npy_events)


def npz_to_arr(npz):
    eve = np.zeros((npz["arr_0"].shape[0], len(npz.keys())))
    for i, k in enumerate(npz.keys()):
        eve[:, i] = npz[k]
    return eve


def npaedat_to_np(events):
    eve = np.zeros((events["timestamp"].shape[0], 4))
    eve[:, 0] = events["timestamp"]
    eve[:, 1] = events["x"]
    eve[:, 2] = events["y"]
    eve[:, 3] = events["polarity"]
    return eve
