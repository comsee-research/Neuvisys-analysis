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

def rf_matching(weights):
    residuals = []
    disparity = []
    for weight in weights:
        res_ref = np.inf
        xmax, ymax = 0, 0
        for x in range(weight.shape[3]):
            for y in range(weight.shape[4]):
                res = weight[:, 0, 0] - np.roll(weight[:, 1, 0], (x, y), axis=(1, 2))

                # fig, ax = plt.subplots(2, 3)
                # fig.suptitle(str(np.sum(np.abs(res))))
                # ax[0, 0].imshow(weight[0, 0, 0])
                # ax[0, 1].imshow(np.roll(weight[0, 1, 0], (x, y), axis=(0, 1)))
                # ax[1, 0].imshow(weight[1, 0, 0])
                # ax[1, 1].imshow(np.roll(weight[1, 1, 0], (x, y), axis=(0, 1)))
                # ax[0, 2].imshow(np.abs(res[0]))
                # ax[1, 2].imshow(np.abs(res[1]))
                
                res = np.sum(res**2)
                if res < res_ref:
                    res_ref = res
                    xmax = x
                    ymax = y
        residuals.append(res_ref)
        disparity.append((xmax, ymax))
    return np.array(residuals), np.array(disparity)

def compute_disparity(spinet, disparity, theta, error, residual, epsi_theta, epsi_error, epsi_residual):
    cnt = 0
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    for i in range(5):
        for j in range(4):
            mask_residual = residual[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_residual
            mask_theta = (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] > np.pi/2+epsi_theta) | (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < np.pi/2-epsi_theta)
            mask_error = error[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_error
            if i != 4 and j != 3:
                axes[j, i].set_title("nb rf:" + str(np.count_nonzero(mask_error & mask_theta & mask_residual)) + "\nmean: " + str(np.mean(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 0][mask_theta & mask_error])))
                axes[j, i].hist(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 0][mask_theta & mask_error & mask_residual], np.arange(-5.5, 6.5), density=True)
            cnt += 1
    
    cnt = 0
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    for i in range(5):
        for j in range(4):
            mask_residual = residual[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_residual
            mask_theta = (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] > epsi_theta) & (theta[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < np.pi-epsi_theta)
            mask_error = error[0, cnt*spinet.l1depth:(cnt+1)*spinet.l1depth] < epsi_error
            if i != 4 and j != 3:
                axes[j, i].set_title("nb rf:" + str(np.count_nonzero(mask_error & mask_theta & mask_residual)) + "\nmean: " + str(np.mean(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 0][mask_theta & mask_error])))
                axes[j, i].hist(disparity[cnt*spinet.l1depth:(cnt+1)*spinet.l1depth, 1][mask_theta & mask_error & mask_residual], np.arange(-5.5, 6.5), density=True)
            cnt += 1
  