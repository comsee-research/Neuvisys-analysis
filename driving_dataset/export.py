#!/usr/bin/env python

'''
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.
'''

from __future__ import print_function
import time, argparse
import Queue
import numpy as np
from view import HDF5Stream, MergedStream
from interfaces.caer import DVS_SHAPE, unpack_data

from aedat_tools import event_address_aedat2, frame_address_aedat2, timestamp_aedat2, write_aedat_header

export_data_vi = {
        'steering_wheel_angle',
        'brake_pedal_status',
        'accelerator_pedal_position',
        'engine_speed',
        'vehicle_speed',
        'windshield_wiper_status',
        'headlamp_status',
        'transmission_gear_position',
        'torque_at_transmission',
        'fuel_level',
        'high_beam_status',
        'ignition_status',
        #'lateral_acceleration',
        'latitude',
        'longitude',
        #'longitudinal_acceleration',
        'odometer',
        'parking_brake_status',
        #'fine_odometer_since_restart',
        'fuel_consumed_since_restart',
    }

export_data_dvs = {
        'dvs_frame',
        'aps_frame',
    }

export_data = export_data_vi.union(export_data_dvs)


def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8

def get_progress_bar():
    try:
        from tqdm import tqdm
    except ImportError:
        print("\n\nNOTE: For an enhanced progress bar, try 'pip install tqdm'\n\n")
        class pbar():
            position=0
            def close(self): pass
            def update(self, increment):
                self.position += increment
                print('\r{}s done...'.format(self.position)),
        def tqdm(*args, **kwargs):
            return pbar()
    return tqdm(total=(tstop-tstart)/1e6, unit_scale=True)

def raster_evts(data):
    _histrange = [(0, v) for v in DVS_SHAPE]
    pol_on = data[:,3] == 1
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            data[pol_on, 2], data[pol_on, 1],
            bins=DVS_SHAPE, range=_histrange)
    img_off, _, _ = np.histogram2d(
            data[pol_off, 2], data[pol_off, 1],
            bins=DVS_SHAPE, range=_histrange)
    return (img_on - img_off).astype(np.int16)


parser = argparse.ArgumentParser()
# parser.add_argument('filename')
parser.add_argument('--tstart', type=int, default=0)
parser.add_argument('--tstop', type=int)
parser.add_argument('--binsize', type=float, default=0.1)
parser.add_argument('--update_prog_every', type=float, default=0.01)
parser.add_argument('--export_aps', type=int, default=1)
parser.add_argument('--export_dvs', type=int, default=1)
parser.add_argument('--out_file', default='')
args = parser.parse_args()

filename = "/home/thomas/Videos/run2/rec1487326422.hdf5"
f_in = HDF5Stream(filename, export_data_vi.union({'dvs'}))
m = MergedStream(f_in)

x_size, y_size = 346, 260
first = 1

aedat_file = "/home/thomas/Desktop/driving.aedat"
with open(aedat_file, "wb") as file:
    write_aedat_header(file)

    tstart = int(m.tmin + 1e6 * args.tstart)
    tstop = (m.tmin + 1e6 * args.tstop) if args.tstop is not None else m.tmax
    m.search(tstart)
    pbar = get_progress_bar()
    sys_ts, t_pre, t_offset, ev_count, pbar_next = 0, 0, 0, 0, 0
    while m.has_data and sys_ts <= tstop*1e-6:
        try:
            sys_ts, d = m.get()
        except Queue.Empty:
            time.sleep(0.01)
            continue
        if not d:
            continue
        if d['etype'] == 'frame_event' and args.export_aps:
            frame = filter_frame(unpack_data(d))
            for y in range(y_size):
                for x in range(x_size):
                    file.write(frame_address_aedat2(x, y, frame[y, x]).tobytes())
                    file.write(timestamp_aedat2(int(1e6 * d["timestamp"])).tobytes())
            continue
        if d['etype'] == 'polarity_event' and args.export_dvs:
            unpack_data(d)
            if (first):
                file.write(event_address_aedat2(x_size-1, y_size-1, 1).tobytes())
                file.write(timestamp_aedat2(d["data"][0][0]).tobytes())
                first = 0
            
            for event in d["data"]:
                file.write(event_address_aedat2(x_size-1-event[1], y_size-1-event[2], event[3]).tobytes())
                file.write(timestamp_aedat2(event[0]).tobytes())
    
        pbar_curr = int((sys_ts - tstart * 1e-6) / args.update_prog_every)
        if pbar_curr > pbar_next:
            pbar.update(args.update_prog_every)
            pbar_next = pbar_curr
    pbar.close()
    print('[DEBUG] sys_ts/tstop', sys_ts, tstop*1e-6)
    m.exit.set()
    
    print('[DEBUG] output done')
    while not m.done.is_set():
        print('[DEBUG] waiting for merger')
        time.sleep(1)
    print('[DEBUG] merger done')
    f_in.join()
    print('[DEBUG] stream joined')
    m.join()
    print('[DEBUG] merger joined')
