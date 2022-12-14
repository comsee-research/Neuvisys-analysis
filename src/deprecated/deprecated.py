#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:12:03 2021

@author: thomas
"""

from bitarray import bitarray
import rosbag
import PySimpleGUI as sg
import json


def write_aedat2_header(aedat_file):
    aedat_file.write(b"#!AER-DAT2.0\r\n")
    aedat_file.write(b"# This is a raw AE data file created by saveaerdat.m\r\n")
    aedat_file.write(
        b"# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n"
    )
    aedat_file.write(b"# Timestamps tick is 1 us\r\n")
    aedat_file.write(b"# End of ASCII Header\r\n")


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

    bits += event_address_aedat2(x_size - 1, y_size - 1, 1)
    bits += timestamp_aedat2(events[0]["timestamp"])

    for event in events:
        bits += event_address_aedat2(
            x_size - 1 - event["x"], y_size - 1 - event["y"], event["polarity"]
        )
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

        for topic, msg, t in bag.read_messages(topics=["/cam0/events"]):
            for n, e in enumerate(msg.events):
                if n == 0:
                    file.write(
                        event_address_aedat2(x_size - 1, y_size - 1, 1).tobytes()
                    )
                    file.write(timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())

                file.write(
                    event_address_aedat2(
                        x_size - 1 - e.x, y_size - 1 - e.y, e.polarity
                    ).tobytes()
                )
                file.write(timestamp_aedat2(int(e.ts.to_nsec() / 1000.0)).tobytes())
        bag.close()


def write_json(directory, gui):
    with open(directory, "w") as file:
        json.dump(gui, file)


def launch_gui(spinet):
    weight_sharing = spinet.weight_sharing
    width = spinet.l1width
    height = spinet.l1height
    depth = spinet.l1depth

    width2 = spinet.l2width
    height2 = spinet.l2height
    depth2 = spinet.l2depth

    sg.theme("DarkAmber")

    layout1 = []

    count = 0
    if weight_sharing:
        for i in range(len(spinet.l1xanchor) * height):
            layout1.append([])

        for i in range(len(spinet.l1xanchor)):
            for j in range(len(spinet.l1yanchor)):
                for col in range(height):
                    for row in range(width):
                        layout1[j * width + row].append(
                            sg.Button("{:>3}".format(str(count)), key="l1" + str(count))
                        )
                        count += 1
                for k in range(width):
                    layout1[j * width + k].append(sg.Text(" "))

        for i in range(1, len(spinet.l1yanchor)):
            layout1.insert(i * height + i - 1, [sg.Text(" ")])
    else:
        for i in range(height):
            layout1.append([])

        for col in range(width):
            for row in range(height):
                layout1[row].append(
                    sg.Button("{:>3}".format(str(count)), key="l1" + str(count))
                )
                count += 1
    layout1.insert(
        0,
        [
            sg.Slider(
                range=(0, depth - 1),
                size=(80, 20),
                orientation="horizontal",
                key="depth1",
                enable_events=True,
            )
        ],
    )
    layout1.append([sg.Button("Save", key="save")])

    count = 0
    layout2 = []

    for i in range(len(spinet.l2xanchor) * height):
        layout2.append([])

    for i in range(len(spinet.l2xanchor)):
        for j in range(len(spinet.l2yanchor)):
            for col in range(height2):
                for row in range(width2):
                    layout2[j * width2 + row].append(
                        sg.Button("{:>3}".format(str(count)), key="l2" + str(count))
                    )
                    count += 1
            for k in range(width2):
                layout2[j * width2 + k].append(sg.Text(" "))
    layout2.insert(
        0,
        [
            sg.Slider(
                range=(0, depth2 - 1),
                orientation="horizontal",
                key="depth2",
                enable_events=True,
            )
        ],
    )

    layout = [[sg.TabGroup([[sg.Tab("Tab 1", layout1), sg.Tab("Tab 2", layout2)]])]]

    # Create the Window
    win = sg.Window(
        "Neuvisys Interface",
        layout,
        default_button_element_size=(1, 1),
        auto_size_buttons=False,
    )

    gui = {"index": 0, "index2": 0, "layer": 0, "layer2": 0, "save": False}

    directory = "/home/alphat/neuvisys-dv/configuration/gui.json"
    x = 0
    y = 0
    index = 0
    x2 = 0
    y2 = 0
    index2 = 0
    while True:
        event, values = win.read()
        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        elif event == "save":
            gui["save"] = True
            write_json(directory, gui)
        elif event == "depth1":
            index = int(x * width * depth + y * depth + values["depth1"])
            gui["index"] = index
            gui["layer"] = int(values["depth1"])
            write_json(directory, gui)
        elif event == "depth2":
            index2 = int(x2 * width2 * depth2 + y2 * depth2 + values["depth2"])
            gui["index2"] = index2
            gui["layer2"] = int(values["depth2"])
            write_json(directory, gui)
        elif "l1" in event:
            x = int(event[2:]) // width
            y = int(event[2:]) % width
            index = int(x * width * depth + y * depth + values["depth1"])
            gui["index"] = index
            gui["layer"] = int(values["depth1"])
            write_json(directory, gui)
        elif "l2" in event:
            x2 = int(event[2:]) // width2
            y2 = int(event[2:]) % width2
            index2 = int(x2 * width2 * depth2 + y2 * depth2 + values["depth2"])
            gui["index2"] = index2
            write_json(directory, gui)
    win.close()
