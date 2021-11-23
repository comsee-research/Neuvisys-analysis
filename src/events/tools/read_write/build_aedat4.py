#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 03:46:47 2020

@author: thomas
"""

import flatbuffers
import re
import sys
import dv.fb.IOHeader
import dv.fb.PacketHeader
import dv.fb.CompressionType
import dv.fb.EventPacket
import dv.fb.Frame
import dv.fb.IMUPacket
import dv.fb.TriggerPacket
import xml.etree.ElementTree as ET
import lz4.frame
import zstd
from dv import Event, Frame, Trigger, IMU


builder = flatbuffers.Builder(1024)

html_info = """<dv version="2.0">
    <node name="outInfo" path="/mainloop/output_file/outInfo/">
        <node name="0" path="/mainloop/output_file/outInfo/0/">
            <attr key="compression" type="string">LZ4</attr>
            <attr key="originalModuleName" type="string">noiseFilter</attr>
            <attr key="originalOutputName" type="string">events</attr>
            <attr key="typeDescription" type="string">Array of events (polarity ON/OFF).</attr>
            <attr key="typeIdentifier" type="string">EVTS</attr>
            <node name="info" path="/mainloop/output_file/outInfo/0/info/">
                <attr key="sizeX" type="int">346</attr>
                <attr key="sizeY" type="int">260</attr>
                <attr key="source" type="string">DAVIS346 [SN: 00000270, USB: 2:3]</attr>
                <attr key="tsOffset" type="long">1584108002921424</attr>
            </node>
        </node>
    </node>
</dv>"""
html_info = builder.CreateString(html_info)

# Header creation
dv.fb.IOHeader.IOHeaderStart(builder)

# Compression choice
dv.fb.IOHeader.IOHeaderAddCompression(
    builder, dv.fb.CompressionType.CompressionType().NONE
)

# Offset for the FileDataTable position
dv.fb.IOHeader.IOHeaderAddDataTablePosition(builder, -1)

# Info string
dv.fb.IOHeader.IOHeaderAddInfoNode(builder, html_info)
# Header end
ioheader = dv.fb.IOHeader.IOHeaderEnd(builder)
builder.FinishSizePrefixed(ioheader)
buf = builder.Output()

builder = flatbuffers.Builder(1024)

# Packet Header
dv.fb.PacketHeader.CreatePacketHeader(builder, 0, 64)

dv.fb.EventPacket.EventPacketStart(builder)
# dv.fb.EventPacket.EventPacketAddEvents(builder, events)
dv.fb.Event.CreateEvent(builder, 1000, 10, 10, 0)

event_packet = dv.fb.EventPacket.EventPacketEnd(builder)
