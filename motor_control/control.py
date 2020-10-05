#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:56:06 2020

@author: thomas
"""

import subprocess
import time

def goto(motor, pos):
    subprocess.run(["/home/thomas/apps/Faulhaber/build/GOTO", str(motor), str(pos)])


goto(0, 10000)
goto(0, 20000)
goto(0, 30000)
goto(0, 20000)
goto(0, 10000)