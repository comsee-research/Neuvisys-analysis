#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:56:06 2020

@author: thomas
"""

import subprocess
import numpy as np
import multiprocessing


def goto(motor, pos):
    subprocess.run(
        ["/home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/build/GOTO", str(motor), str(pos)]
    )

def test():
    positions = np.arange(0, 100000, 10000)
    
    p = multiprocessing.Pool()
    
    for position in positions:
        p.starmap(goto, [(0, position), (1, position)])
    
    for position in positions[::-1]:
        p.starmap(goto, [(0, position), (1, position)])
