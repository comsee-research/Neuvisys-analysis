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
from aedat_tools.aedat_tools import load_params

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

def spike_plots_simple_cells(spinet, neuron_id):
    plt.figure()
    plt.xlabel("time (µs)")
    plt.ylabel("neurons")
    
    indices = spinet.simple_cells[neuron_id].inhibition_connections
    colors1 = ['C{}'.format(i) for i in range(len(indices)+1)]
    
    eveplot = []
    for i in indices + [neuron_id]:
        eveplot.append(spinet.sspikes[i][spinet.sspikes[i] != neuron_id])
        
    plt.eventplot(eveplot, colors=colors1)

def spike_plots_complex_cells(spinet, neuron_id):
    plt.figure()
    plt.xlabel("time (µs)")
    plt.ylabel("neurons")
    
    indices = spinet.complex_cells[neuron_id].inhibition_connections
    colors1 = ['C{}'.format(i) for i in range(len(indices)+1)]
    
    eveplot = []
    for i in indices + [neuron_id]:
        eveplot.append(spinet.cspikes[i][spinet.cspikes[i] != neuron_id])
        
    plt.eventplot(eveplot, colors=colors1)
    
def crosscorrelogram(spinet):
    indices = []
    for i in range(spinet.l1width):
        for j in range(spinet.l1height):
            for k in range(spinet.l1depth):
               indices.append(spinet.layout1[i, j, k])
               
    test = spinet.sspikes[indices, :]
    
    bin_size = 100000 #µs
    nb_bins = 19
    
    ref_id = 10
    
    strain_ref = test[ref_id]
    
    for i in range(100):
        strain_tar = test[i]
        
        a = np.zeros(nb_bins-1)
        for tmstp in strain_ref[strain_ref != 0]:
            bins = np.linspace(tmstp-bin_size, tmstp+bin_size, nb_bins)
            a += np.histogram(strain_tar, bins)[0]
    
        plt.figure()
        plt.title("Crosscorrelogram reference neuron "+str(ref_id)+" against neuron "+str(i))
        plt.ylabel("count")
        plt.xlabel("time delay (ms)")
        plt.plot(np.linspace(-bin_size/1e3, bin_size/1e3, nb_bins-1), a)
        plt.savefig("/home/alphat/Desktop/report/crosscorr_inh/"+str(i))

def orientation_norm_length(spike_vector, angles):
    return np.abs(np.sum(spike_vector * np.exp(2j*angles)) / np.sum(spike_vector))
        
def direction_norm_length(spike_vector, angles):
    return np.abs(np.sum(spike_vector * np.exp(1j*angles)) / np.sum(spike_vector))
    
def direction_selectivity(spike_vector):
    return (np.max(spike_vector) - spike_vector[(np.argmax(spike_vector)+8)%16]) / np.max(spike_vector)
        
def orientation_selectivity(spike_vector):
    return (np.max(spike_vector) - spike_vector[(np.argmax(spike_vector)+4)%8]) / np.max(spike_vector)

def weight_centroid(weight):
    weight = weight[0] + weight[1]
    weight /= np.max(weight)
    (X, Y) = np.indices((weight.shape[0], weight.shape[1]))
    x_coord = (X * weight).sum() / weight.sum()
    y_coord = (Y * weight).sum() / weight.sum()
    return x_coord, y_coord

def centroids(spinet):
    weights = []
    if spinet.weight_sharing == "full":
        weights = [neuron.weights for neuron in spinet.simple_cells[0:spinet.l1depth]]
    elif spinet.weight_sharing == "patch":
        for i in range(0, spinet.nb_simple_cells, spinet.l1depth*spinet.l1width*spinet.l1height):
            weights += [neuron.weights for neuron in spinet.simple_cells[i:i+spinet.l1depth]]
    else:
        weights = [neuron.weights for neuron in spinet.simple_cells]
    
    l_c, r_c = [], []
    for weight in weights:
        l_c.append(weight_centroid(weight[:, 0, 0]))
        r_c.append(weight_centroid(weight[:, 1, 0]))
    return np.array(l_c), np.array(r_c)