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

from itertools import combinations
from PIL import Image
from natsort import natsorted

from utils.utils import load_params

def inhibition_boxplot(directory):
    dist = []
    for network_id in range(2):
        direc = directory+"/images/"+str(network_id)+"/"
        images = natsorted(os.listdir(direc))
        distances = []
    
        for i in range(0, 3536, 4):
            vectors = []
            for j in range(4):
                vectors.append(np.asarray(Image.open(direc + images[i+j]), dtype=float) / 255)
            
            for combination in combinations(vectors, 2):
                distances.append(np.sum((combination[0] - combination[1])**2) / 100)
        dist.append(distances)
    
    fig, axes = plt.subplots()
    
    fig.subplots_adjust(left=0.3, right=0.8)
    
    axes.set_title('', y=-0.1, fontsize=14)
    axes.set_ylabel("Squared Euclidean distance", fontsize=14)
    axes.boxplot([dist[1], dist[0]])
    axes.xaxis.set_ticklabels(["No inhibition", "Inhibition"], fontsize=14)
    plt.savefig("boxplots.pdf", bbox_inches="tight")

def params_network(directory):
    params = []
    for entry in natsorted(os.listdir(directory+"weights/")):
        if entry.endswith(".json"):
            params.append(load_params(directory+"weights/"+entry))
    return pd.DataFrame(params)

def networks_stats(nb_networks, directory):
    df = []
    for i in range(nb_networks):
        df.append(load_params(directory+"network_"+str(i)+"/configs/config.json"))
        
        net_param = params_network(directory+"network_"+str(i)+"/")
        df[i]["count_spike"] = net_param["count_spike"].mean()
        df[i]["learning_decay"] = net_param["learning_decay"].mean()
        df[i]["threshold"] = net_param["threshold"].mean()
    return pd.DataFrame(df)

####

directory = "/home/thomas/neuvisys-dv/configuration/Run1/"
df = params_network(directory+"network_0/")
df_networks = networks_stats(10, directory)