#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:10:06 2020

@author: thomas
"""

import os
import scipy.io as sio
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

def plot_gabor_image(neuron, est_basis, error, path, count):
    gabor = -1 * est_basis[:, count].reshape(10, 10, order="F")
    fig, axes = plt.subplots(2, 1)
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].imshow(mpimg.imread(neuron.weight_images[0]))
    axes[1].imshow(gabor)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_gabors(spinet, mu, sigma, lambd, phase, theta, error, est_basis, dest):
    cnt = 0
    indices = np.arange(0, spinet.nb_simple_cells, spinet.l1width * spinet.l1height * spinet.l1depth)
    if spinet.weight_sharing:
        for i, ind in enumerate(indices):
            for neuron in spinet.simple_cells[ind:ind+spinet.l1depth]:
                path = dest + str(cnt) + "_" + str(round(error[0, cnt], 2)) + ".png"
                plot_gabor_image(neuron, est_basis, error, dest, cnt)
                neuron.add_gabor(path, mu[0, cnt], sigma[0, cnt], lambd[0, cnt], phase[0, cnt], theta[0, cnt], error[0, cnt])
                cnt += 1
            for j, neuron in enumerate(spinet.simple_cells[ind+spinet.l1depth:ind+spinet.l1width*spinet.l1height*spinet.l1depth]):
                c = i * spinet.l1depth + j % 144
                path = dest + str(c) + "_" + str(round(error[0, c], 2)) + ".png"
                neuron.add_gabor(path, mu[0, c], sigma[0, c], lambd[0, c], phase[0, c], theta[0, c], error[0, c])
        
        # for i in range(0, spinet.nb_simple_cells, spinet.l1width * spinet.l1height * spinet.l1depth):
        #     for neuron in spinet.simple_cells[i:i+spinet.l1depth]:
        #         path = dest + str(count) + "_" + str(round(error[0, count], 2)) + ".png"
        #         neuron.add_gabor(path, mu[0, count], sigma[0, count], lambd[0, count], phase[0, count], theta[0, count], error[0, count])
        #         plot_gabor_image(neuron, est_basis, error, dest, count)
        #         count += 1
    else:
        for neuron in spinet.simple_cells:
            neuron.add_gabor(path, mu[0, cnt], sigma[0, cnt], lambd[0, cnt], phase[0, cnt], theta[0, cnt], error[0, cnt])
            plot_gabor_image(neuron, est_basis, error, dest, cnt)
            cnt += 1
                

def plot_histogram(theta, error, err_thresh, dest):
    plt.figure()
    plt.hist(theta[error < err_thresh] * 180 / np.pi, bins=10)
    plt.xticks(np.arange(0, 181, 30))
    plt.xlabel("Phase (degree)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig(dest, bbox_inches="tight")
    plt.close()
    
def plot_polar_chart(depth, bins, theta, error, err_thresh, dest):
    fig, axes = plt.subplots(3, 3, subplot_kw=dict(projection='polar'))
    for i in range(3):
        for j in range(3):
            sub_theta = theta[(i*3+j)*depth:(i*3+j+1)*depth]
            sub_error = error[(i*3+j)*depth:(i*3+j+1)*depth]
            hist, _ = np.histogram(sub_theta[sub_error < err_thresh], bins, range=(0, np.pi))
            the1 = np.linspace(0.0, np.pi, hist.size)
            the1 = np.concatenate((the1, the1+np.pi))
            hist = np.concatenate((hist, np.flipud(hist)))
            
            axes[j, i].plot(the1, hist, "r", linewidth=2)
            axes[j, i].set_xticks(np.arange(0, np.pi+0.0001, np.pi/4.0))
            axes[j, i].set_theta_zero_location("W")
            axes[j, i].set_theta_direction(-1)

    plt.savefig(dest+"region_histogram.pdf", bbox_inches="tight")
    
    plt.figure()
    ax = plt.subplot(111, polar=True)
    hist, _ = np.histogram(theta[error < err_thresh], bins, range=(0, np.pi))
    the1 = np.linspace(0.0, np.pi, hist.size)
    the2 = np.linspace(np.pi, 2*np.pi, hist.size)
    ax.plot(the1, hist, "r")
    ax.set_xticks(np.arange(0, np.pi+0.0001, np.pi/6.0))
    ax.set_thetamax(180)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_ylabel("orientation (°)")

    plt.savefig(dest+"total_histogram.pdf", bbox_inches="tight")
        
def error_percentage(theta, error, max_error, dest):
    count = []
    x = np.arange(0, max_error, 0.01)
    for err in x:
        count.append(theta[error < err].size / theta.size)

    plt.figure()
    plt.plot(x, count)
    plt.title("Proportion of accepted gabors function of the error threshold")
    plt.xlabel("error threshold")
    plt.ylabel("proportion of accepted gabors (%)")
    plt.savefig(dest+"error_proportion", bbox_inches="tight")
    
def create_gabor_basis(spinet, bins):
    mu = sio.loadmat(spinet.path+"gabors/data/mu.mat")["mu_table"]
    sigma = sio.loadmat(spinet.path+"gabors/data/sigma.mat")["sigma_table"]
    lambd = sio.loadmat(spinet.path+"gabors/data/lambda.mat")["lambda_table"]
    phase = sio.loadmat(spinet.path+"gabors/data/phase.mat")["phase_table"]
    theta = sio.loadmat(spinet.path+"gabors/data/theta.mat")["theta_table"]
    error = sio.loadmat(spinet.path+"gabors/data/error.mat")["error_table"]
    est_basis = sio.loadmat(spinet.path+"gabors/data/EstBasis.mat")["EstBasis"]
    
    plot_gabors(spinet, mu, sigma, lambd, phase, theta, error, est_basis, spinet.path+"gabors/figures/")
    plot_polar_chart(spinet.l1depth, bins, theta[0], error[0], 5, spinet.path+"gabors/hists/")
    error_percentage(theta[0], error[0], 20, spinet.path+"gabors/hists/")