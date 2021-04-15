#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:10:06 2020

@author: thomas
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_gabor_image(neuron, est_basis, error, path, count, side):
    gabor = -1 * est_basis[:, count].reshape(10, 10, order="F")
    fig, axes = plt.subplots(2, 1)
    axes[0].axis("off")
    axes[1].axis("off")
    if side == "left":
        axes[0].imshow(mpimg.imread(neuron.weight_images[0]))
    elif side == "right":
        axes[0].imshow(mpimg.imread(neuron.weight_images[1]))
    axes[1].imshow(gabor)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gabors(spinet, mu, sigma, lambd, phase, theta, error, est_basis, dest, side):
    cnt = 0
    indices = np.arange(
        0,
        spinet.nb_simple_cells,
        spinet.conf["L1Depth"] * spinet.conf["L1Width"] * spinet.conf["L1Height"],
    )
    if spinet.conf["SharingType"] == "full":
        for i in range(spinet.conf["L1Depth"]):
            path = dest + "{0:.2f}".format(error[0, i]) + str(i) + "_" + side + ".png"
            plot_gabor_image(spinet.simple_cells[i], est_basis, error, path, i, side)
        for neuron in spinet.simple_cells:
            x, y, z = neuron.params["position"]
            path = dest + "{0:.2f}".format(error[0, z]) + str(z) + "_" + side + ".png"
            neuron.add_gabor(
                path,
                mu[0, z],
                sigma[0, z],
                lambd[0, z],
                phase[0, z],
                theta[0, z],
                error[0, z],
            )
    elif spinet.conf["SharingType"] == "patch":
        for i, ind in enumerate(indices):
            for neuron in spinet.simple_cells[ind : ind + spinet.conf["L1Depth"]]:
                path = (
                    dest
                    + str(cnt)
                    + "_{0:.2f}".format(error[0, cnt])
                    + "_"
                    + side
                    + ".png"
                )
                plot_gabor_image(neuron, est_basis, error, path, cnt, side)
                neuron.add_gabor(
                    path,
                    mu[0, cnt],
                    sigma[0, cnt],
                    lambd[0, cnt],
                    phase[0, cnt],
                    theta[0, cnt],
                    error[0, cnt],
                )
                cnt += 1
            for j, neuron in enumerate(
                spinet.simple_cells[
                    ind
                    + spinet.conf["L1Depth"] : ind
                    + spinet.conf["L1Depth"]
                    * spinet.conf["L1Width"]
                    * spinet.conf["L1Height"]
                ]
            ):
                c = i * spinet.conf["L1Depth"] + j % spinet.conf["L1Depth"]
                path = (
                    dest + str(c) + "_{0:.2f}".format(error[0, c]) + "_" + side + ".png"
                )
                neuron.add_gabor(
                    path,
                    mu[0, c],
                    sigma[0, c],
                    lambd[0, c],
                    phase[0, c],
                    theta[0, c],
                    error[0, c],
                )
    else:
        for neuron in spinet.simple_cells:
            neuron.add_gabor(
                path,
                mu[0, cnt],
                sigma[0, cnt],
                lambd[0, cnt],
                phase[0, cnt],
                theta[0, cnt],
                error[0, cnt],
            )
            plot_gabor_image(neuron, est_basis, error, dest, cnt, side)
            cnt += 1


def plot_polar_chart(depth, nb_ticks, theta, error, err_thresh, dest):
    fig, axes = plt.subplots(3, 3, subplot_kw=dict(projection="polar"))
    for i in range(3):
        for j in range(3):
            sub_theta = theta[(i * 3 + j) * depth : (i * 3 + j + 1) * depth]
            sub_error = error[(i * 3 + j) * depth : (i * 3 + j + 1) * depth]
            hist = compute_histogram(
                sub_theta[sub_error < err_thresh] * 180 / np.pi, 180, nb_ticks
            )

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


def create_gabor_basis(spinet, side, nb_ticks):
    mu = sio.loadmat(spinet.path + "gabors/data/" + side + "/mu.mat")["mu_table"]
    sigma = sio.loadmat(spinet.path + "gabors/data/" + side + "/sigma.mat")[
        "sigma_table"
    ]
    lambd = sio.loadmat(spinet.path + "gabors/data/" + side + "/lambda.mat")[
        "lambda_table"
    ]
    phase = sio.loadmat(spinet.path + "gabors/data/" + side + "/phase.mat")[
        "phase_table"
    ]
    theta = sio.loadmat(spinet.path + "gabors/data/" + side + "/theta.mat")[
        "theta_table"
    ]
    error = sio.loadmat(spinet.path + "gabors/data/" + side + "/error.mat")[
        "error_table"
    ]
    est_basis = sio.loadmat(spinet.path + "gabors/data/" + side + "/EstBasis.mat")[
        "EstBasis"
    ]

    theta = (theta + np.pi / 2) % np.pi

    plot_gabors(
        spinet,
        mu,
        sigma,
        lambd,
        phase,
        theta,
        error,
        est_basis,
        spinet.path + "gabors/figures/",
        side,
    )
    plot_polar_chart(
        spinet.l1depth, nb_ticks, theta[0], error[0], 20, spinet.path + "gabors/hists/"
    )
    error_percentage(theta[0], error[0], 20, spinet.path + "gabors/hists/")

    return mu, sigma, lambd, phase, theta, error


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

    for i in range(spinet.nb_complex_cells):
        complex_cell = spinet.complex_cells[i]
        ox, oy, oz = complex_cell.offset

        orientations = []
        strengths = []
        maximum = np.max(complex_cell.weights)
        for connection in complex_cell.in_connections:
            simple_cell = spinet.simple_cells[connection]
            xs, ys, zs = simple_cell.position
            strengths.append(complex_cell.weights[xs - ox, ys - oy, zs] / maximum)
            orientations.append(simple_cell.orientation * 180 / np.pi)

        hists_o.append(compute_histogram(orientations, 180, 8, strengths))

        if i % spinet.l2depth == 0:
            hists_ro.append(compute_histogram(orientations, 180, 8))

    return np.array(hists_o), np.array(hists_ro)


def plot_preferred_orientations(spinet, hists_o, hists_ro):
    i = 0
    for hist_o in hists_o:
        circular_plot(
            "complex cell (" + str(i) + ") prefered orientation", hist_o, 180, 8
        )
        plt.savefig(spinet.path + "figures/complex_orientations/" + str(i))
        i += 1

    i = 0
    for hist_ro in hists_ro:
        circular_plot("Histogram of a region orientations", hist_ro, 180, 8)
        plt.savefig(spinet.path + "figures/complex_orientations/" + "r_" + str(i))
        i += 1
