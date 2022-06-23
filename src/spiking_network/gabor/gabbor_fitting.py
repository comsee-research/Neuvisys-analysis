#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:10:06 2020

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_gabor_basis(spinet, gabor_fit_res, camera, nb_ticks, error_thresh):
    est_basis = gabor_fit_res[0][camera]
    mu = gabor_fit_res[2][camera]
    lambd = gabor_fit_res[3][camera]
    sigma = gabor_fit_res[4][camera]
    theta = gabor_fit_res[5][camera]
    phase = gabor_fit_res[6][camera]
    error = gabor_fit_res[7][camera]

    theta = (theta + np.pi / 2) % np.pi

    plot_gabors(spinet, mu, sigma, lambd, phase, theta, error, est_basis, spinet.path + "images/0/gabors/", camera)
    plot_polar_chart(spinet, nb_ticks, theta, error, error_thresh, spinet.path + "figures/0/")
    error_percentage(theta, error, error_thresh, spinet.path + "figures/0/")

    return mu, sigma, lambd, phase, theta, error


def plot_gabor_image(neuron, gabor, path, camera):
    gabor = -1 * gabor.reshape(10, 10)
    fig, axes = plt.subplots(2, 1)
    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].imshow(mpimg.imread(neuron.weight_images[camera]))
    axes[1].imshow(gabor)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gabors(spinet, mu, sigma, lambd, phase, theta, error, est_basis, dest, camera):
    if spinet.conf["sharingType"] == "full":
        for i in range(spinet.l_shape[0, 2]):
            path = dest + str(i) + "_{0:.2f}".format(error[i]) + str(i) + "_" + str(camera) + ".png"
            plot_gabor_image(spinet.neurons[0][i], est_basis[:, i], path, camera)
        for neuron in spinet.neurons[0]:
            x, y, z = neuron.params["position"]
            path = dest + "{0:.2f}".format(error[z]) + str(z) + "_" + str(camera) + ".png"
            neuron.add_gabor(path, mu[0, z], sigma[0, z], lambd[z], phase[z], theta[z], error[z])
    elif spinet.conf["SharingType"] == "patch":
        indices = np.arange(0, len(spinet.neurons[0]),
                            spinet.l_shape[0, 0] * spinet.l_shape[0, 1] * spinet.l_shape[0, 2])
        for i, ind in enumerate(indices):
            for j, neuron in enumerate(spinet.neurons[0][ind: ind + spinet.l_shape[0, 2]]):
                path = dest + str(j) + "_{0:.2f}".format(error[j]) + "_" + str(camera) + ".png"
                plot_gabor_image(neuron, est_basis[:, j], path, camera)
                neuron.add_gabor(path, mu[0, j], sigma[0, j], lambd[j], phase[j], theta[j], error[j])
                j += 1
            for k, neuron in enumerate(spinet.neurons[0][
                                       ind + spinet.l_shape[0, 2]: ind + spinet.l_shape[0, 0] * spinet.l_shape[0, 1] *
                                                                 spinet.l_shape[0, 2]]):
                index = i * spinet.l_shape[0, 2] + k % spinet.l_shape[0, 2]
                path = dest + str(index) + "_{0:.2f}".format(error[index]) + "_" + str(camera) + ".png"
                neuron.add_gabor(path, mu[0, index], sigma[0, index], lambd[index], phase[index], theta[index],
                                 error[index])
    else:
        for i, neuron in enumerate(spinet.neurons[0]):
            path = dest + str(i) + "_{0:.2f}".format(error[i]) + "_" + str(camera) + ".png"
            neuron.add_gabor(path, mu[0, i], sigma[0, i], lambd[i], phase[i], theta[i], error[i])
            plot_gabor_image(neuron, est_basis[:, i], dest, camera)


def plot_polar_chart(spinet, nb_ticks, theta, error, err_thresh, dest):
    depth = spinet.l_shape[0, 2]
    if spinet.conf["sharingType"] == "patch":
        patches = spinet.p_shape
        fig, axes = plt.subplots(patches[0], patches[1], subplot_kw=dict(projection="polar"))
        for i in range(patches[0]):
            for j in range(patches[1]):
                sub_theta = theta[(i * patches[0] + j) * depth: (i * patches[0] + j + 1) * depth]
                sub_error = error[(i * patches[0] + j) * depth: (i * patches[0] + j + 1) * depth]
                hist = compute_histogram(sub_theta[sub_error < err_thresh] * 180 / np.pi, 180, nb_ticks)

                x = list(np.arange(0, np.pi, np.pi / nb_ticks))
                x.append(180)
                axes[j, i].plot(x, hist, "r")
                axes[j, i].set_xticks(np.arange(0, np.pi + 0.1, np.pi / nb_ticks))
                axes[j, i].set_thetamax(180)
                axes[j, i].set_theta_zero_location("N")
                axes[j, i].set_theta_direction(-1)
        plt.savefig(dest + "region_histogram.pdf", bbox_inches="tight")

    hist = compute_histogram(theta[error < err_thresh] * 180 / np.pi, 180, nb_ticks)
    circular_plot("plot", hist, 180, nb_ticks)
    plt.savefig(dest + "total_histogram.pdf", bbox_inches="tight")


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
    plt.savefig(dest + "error_proportion", bbox_inches="tight")


def compute_histogram(directions, thet_max, nb_ticks=8, weights=None):
    bins = list(np.arange((180 / nb_ticks) / 2, thet_max, 180 / nb_ticks))
    bins.insert(0, 0)
    bins.append(thet_max)
    hist, _ = np.histogram(directions, bins, weights=weights)
    hist[0] = hist[0] + hist[-1]
    hist[-1] = hist[0]
    return hist


def circular_plot(title, hist, thet_max, nb_ticks=8):
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_title(title)

    x = list(np.arange(0, (thet_max / 180) * np.pi, np.pi / nb_ticks))
    x.append(0) if thet_max == 360 else x.append(np.pi)
    ax.plot(x, hist, "r")
    ax.set_thetamax(thet_max)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def hists_preferred_orientations(spinet):
    hists_o = []
    hists_ro = []

    for i in range(len(spinet.neurons[1])):
        complex_cell = spinet.neurons[1][i]
        ox, oy, oz = complex_cell.params["offset"]

        orientations = []
        strengths = []
        maximum = np.max(complex_cell.weights)
        for connection in complex_cell.params["in_connections"]:
            simple_cell = spinet.neurons[0][connection]
            xs, ys, zs = simple_cell.params["position"]
            strengths.append(complex_cell.weights[xs - ox, ys - oy, zs] / maximum)
            orientations.append(simple_cell.orientation * 180 / np.pi)

        hists_o.append(compute_histogram(orientations, 180, 8, strengths))

        if i % spinet.l_shape[1, 2] == 0:
            hists_ro.append(compute_histogram(orientations, 180, 8))

    return np.array(hists_o), np.array(hists_ro)


def plot_preferred_orientations(spinet, hists_o, hists_ro):
    i = 0
    for hist_o in hists_o:
        circular_plot("complex cell (" + str(i) + ") preferred orientation", hist_o, 180, 8)
        plt.savefig(spinet.path + "figures/1/preferred_orientation/" + str(i) + "")
        i += 1

    i = 0
    for hist_ro in hists_ro:
        circular_plot("Histogram of a region orientations", hist_ro, 180, 8)
        plt.savefig(spinet.path + "figures/1/" + "r_" + str(i))
        i += 1
