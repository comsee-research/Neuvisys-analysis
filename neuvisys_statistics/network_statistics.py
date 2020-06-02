#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:25 2020

@author: thomas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils import load_params

def params_network(directory):
    params = []
    for entry in os.scandir(directory+"weights/"):
        if entry.path.endswith(".json"):
            params.append(load_params(entry.path))
    return pd.DataFrame(params)

def average_networks(nb_networks, directory):
    averages = {"count_spike": [], "learning_decay": [], "threshold": []}
    for i in range(nb_networks):
        df = params_network(directory+"network_"+str(i)+"/")
        averages["count_spike"].append(df["count_spike"].mean())
        averages["learning_decay"].append(df["learning_decay"].mean())
        averages["threshold"].append(df["threshold"].mean())
    return pd.DataFrame(averages)


directory = "/home/thomas/neuvisys-dv/configuration/Run3/"
df = average_networks(10, directory)