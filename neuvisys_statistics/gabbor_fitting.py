#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:10:06 2020

@author: thomas
"""

import os
import scipy.io as sio
import numpy as np
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations

from utils import load_params

def plot_gabors():
    for image_id in range(3536):
        gabor = gabor = -1 * est_basis[:, image_id].reshape(10, 10, order="F")
        image = np.moveaxis(np.concatenate((np.load(files[image_id]), np.zeros((1, 10, 10))), axis=0), 0, 2)
        
        fig, axes = plt.subplots(2, 1)
        axes[0].axis('off')
        axes[1].axis('off')
        axes[0].imshow(image)
        axes[1].imshow(gabor)
        plt.savefig("/home/thomas/Desktop/gabors/" + str(image_id) + "_" + str(round(error[0, image_id], 2)) + ".png", bbox_inches="tight")
        plt.close(fig)
        
def plot_gabor(image_id):
    gabor = -1 * est_basis[:, image_id].reshape(10, 10, order="F")
    image = np.moveaxis(np.concatenate((np.load(files[image_id]), np.zeros((1, 10, 10))), axis=0), 0, 2)
    
    fig, axes = plt.subplots(2, 1)
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].imshow(image)
    axes[1].imshow(gabor, cmap=cm.RdYlGn)
    
def plot_histogram():
    plt.figure()
    plt.hist(theta[0, error[0] < 4] * 180 / np.pi, bins=25)
    plt.xticks(np.arange(0, 181, 30))
    plt.xlabel("Phase (degree)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig("histogram.pdf", bbox_inches="tight")
    
# for i, file in enumerate(files):
#     for synapse in range(network_params["NEURON_SYNAPSES"]):
#         weights = np.load(file)

#         weight = (weights[0] - weights[1]).flatten("F")
#         basis[0:100, i] = weight
#         basis[100:200, i] = weight
# sio.savemat("/home/thomas/Desktop/GaborFitting2D/weights.mat", {"data": basis})

directory = "/home/thomas/neuvisys-dv/configuration/"
network_id = 0

basis = np.zeros((200, 3536))

files = natsorted([directory+"weights/"+str(network_id)+"/"+f for f in os.listdir(directory+"weights/"+str(network_id)+"/") if f.endswith(".npy")])
network_params = load_params(directory+"configs/"+str(network_id)+".json")

mu = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/mu.mat")["mu_table"]
sigma = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/sigma.mat")["sigma_table"]
lambd = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/lambda.mat")["lambda_table"]
phase = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/phase.mat")["phase_table"]
theta = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/theta.mat")["theta_table"]
error = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/error.mat")["error_table"]

est_basis = sio.loadmat("/home/thomas/Desktop/GaborFitting2D/mat/EstBasis.mat")["EstBasis"]

gabors = []
images = []
weigs = []
for image_id in range(3536):
    # gabors.append(cv.getGaborKernel((10, 10), sigma=29, theta=theta[0, image_id], lambd=lambd[0, image_id], gamma=1, psi=phase[0, image_id]))
    gabors.append(est_basis[:, image_id].reshape(10, 10))
    
    image = np.moveaxis(np.concatenate((np.load(files[image_id]), np.zeros((1, 10, 10))), axis=0), 0, 2)
    images.append(image)

plot_histogram()

####

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