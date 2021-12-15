#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:47:43 2020

@author: thomas
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from fpdf import FPDF
from natsort import natsorted


def pdf_simple_cell(spinet, layer, camera):
    pdf = FPDF(
        "P", "mm", (11 * spinet.conf["L1Width"], 11 * spinet.conf["L1Height"] * spinet.conf["Neuron1Synapses"],),
    )
    pdf.add_page()

    pos_x = 0
    pos_y = 0
    for neuron in spinet.neurons[0]:
        x, y, z = neuron.params["position"]
        if z == layer:
            for i in range(spinet.conf["neuron1Synapses"]):
                pos_x = x * 11
                pos_y = y * spinet.conf["neuron1Synapses"] * 11 + i * 11
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_simple_cell_left_right_combined(spinet, layer):
    pdf = FPDF(
        "P", "mm", (11 * spinet.conf["L1Width"], 24 * spinet.conf["L1Height"] * spinet.conf["Neuron1Synapses"],),
    )
    pdf.add_page()

    pos_x = 0
    pos_y = 0
    for neuron in spinet.neurons[0]:
        x, y, z = neuron.params["position"]
        if z == layer:
            for i in range(spinet.conf["neuron1Synapses"]):
                pos_x = 11 * x
                pos_y = 24 * y * spinet.conf["neuron1Synapses"] + i * 11
                pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
                pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def pdf_layers(spinet, rows, cols, nb_synapses, nb_layers):
    images = natsorted(os.listdir(spinet.path + "images/simple_cells/"))
    pdf = FPDF("P", "mm", (cols * 11, rows * 11 * nb_layers))
    pdf.add_page()

    count = 0
    for i in range(cols):
        for j in range(rows):
            for l in range(nb_layers):
                pdf.image(
                    spinet.path + "images/simple_cells/" + images[count],
                    x=i * 11,
                    y=j * 11 * nb_layers + l * 10.4,
                    w=10,
                    h=10,
                )
                count += nb_synapses
    return pdf


def pdf_weight_sharing(spinet, camera):
    if spinet.conf["sharingType"] == "full":
        side = int(np.sqrt(spinet.l_shape[0, 2]))
        xpatch = 1
        ypatch = 1
    elif spinet.conf["sharingType"] == "patch":
        side = int(np.sqrt(spinet.l_shape[0, 2]))
        xpatch = spinet.p_shape[0, 0].shape[0]
        ypatch = spinet.p_shape[0, 1].shape[0]
    pdf = FPDF("P", "mm", (11 * xpatch * side + (xpatch - 1) * 10, 11 * ypatch * side + (ypatch - 1) * 10,), )
    pdf.add_page()

    pos_x = 0
    pos_y = 0
    shift = np.arange(spinet.l_shape[0, 2]).reshape((side, side))
    if spinet.conf["sharingType"] == "full":
        cell_range = range(spinet.l_shape[0, 2])
    elif spinet.conf["sharingType"] == "patch":
        cell_range = range(
            0, len(spinet.neurons[0]), spinet.l_shape[0, 2] * spinet.l_shape[0, 0] * spinet.l_shape[0, 1],
        )
    for i in cell_range:
        for neuron in spinet.neurons[0][i: i + spinet.l_shape[0, 2]]:
            x, y, z = neuron.params["position"]
            pos_x = (
                    (x // spinet.l_shape[0, 0]) * side * 11
                    + np.where(shift == z)[0][0] * 11
                    + (x // spinet.l_shape[0, 0]) * 10
            )  # patch size + weight sharing shift + patch padding
            pos_y = (
                    (y // spinet.l_shape[0, 1]) * side * 11
                    + np.where(shift == z)[1][0] * 11
                    + (y // spinet.l_shape[0, 1]) * 10
            )
            pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_weight_sharing_left_right_combined(spinet):
    side = int(np.sqrt(spinet.conf["L1Depth"]))
    pdf = FPDF(
        "P",
        "mm",
        (
            11 * len(spinet.conf["L1XAnchor"]) * side + (len(spinet.conf["L1XAnchor"]) - 1) * 10,
            24 * len(spinet.conf["L1YAnchor"]) * side + (len(spinet.conf["L1YAnchor"]) - 1) * 10,
        ),
    )
    pdf.add_page()

    pos_x = 0
    pos_y = 0
    shift = np.arange(spinet.conf["L1Depth"]).reshape((side, side))
    for i in range(
            0, spinet.nb_simple_cells, spinet.conf["L1Depth"] * spinet.conf["L1Width"] * spinet.conf["L1Height"],
    ):
        for neuron in spinet.neurons[0][i: i + spinet.conf["L1Depth"]]:
            x, y, z = neuron.params["position"]
            pos_x = (
                    (x // spinet.conf["L1Width"]) * side * 11
                    + np.where(shift == z)[0][0] * 11
                    + (x // spinet.conf["L1Width"]) * 10
            )  # patch size + weight sharing shift (x2) + patch padding
            pos_y = (
                    (y // spinet.conf["L1Height"]) * side * 24
                    + np.where(shift == z)[1][0] * 24
                    + (y // spinet.conf["L1Height"]) * 10
            )
            pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
            pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def display_motor_cell(spinet):
    for i, motor_cell in enumerate(spinet.motor_cells):
        heatmap = np.mean(motor_cell.weights, axis=2)

        plt.figure()
        plt.matshow(heatmap)
        plt.savefig(spinet.path + "figures/motor_figures/" + str(i), bbox_inches="tight")


def pdf_complex_cell(spinet, zcell, layer):
    pdf = FPDF(
        "P",
        "mm",
        (
            spinet.p_shape[layer, 0].shape[0] * spinet.l_shape[layer, 0] * 11 + spinet.p_shape[layer, 0].shape[0] * 11,
            spinet.p_shape[layer, 1].shape[0] * spinet.l_shape[layer, 1] * spinet.n_shape[layer, 2] * 11
            + spinet.p_shape[layer, 1].shape[0] * 11
            + spinet.p_shape[layer, 1].shape[0] * 10,
        ),
    )
    pdf.add_page()

    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "")

    for c, complex_cell in enumerate(spinet.neurons[layer]):
        xc, yc, zc = complex_cell.params["position"]
        ox, oy, oz = complex_cell.params["offset"]

        heatmap = np.zeros((spinet.n_shape[layer, 0], spinet.n_shape[layer, 1]))
        heatmap_rf = np.zeros((120, 120, 3))

        if zc == zcell:
            maximum = np.max(complex_cell.weights)
            for z, k in enumerate(sort_connections(spinet, complex_cell, spinet.n_shape[layer, 2])):
                for i in range(ox, ox + spinet.l_shape[layer, 0]):
                    for j in range(oy, oy + spinet.l_shape[layer, 1]):
                        simple_cell = spinet.neurons[0][spinet.layout[layer - 1][i, j, k]]
                        xs, ys, zs = simple_cell.params["position"]

                        weight_sc = complex_cell.weights[xs - ox, ys - oy, k] / maximum
                        img = weight_sc * np.array(Image.open(simple_cell.weight_images[0]))
                        path = (
                                spinet.path
                                + "figures/1/tmp/"
                                + str(c)
                                + "_simple_"
                                + str(spinet.layout[layer - 1][i, j, k])
                                + ".png"
                        )
                        Image.fromarray(img.astype("uint8")).save(path)

                        heatmap[ys - oy, xs - ox] += weight_sc
                        if np.argmax(complex_cell.weights[ys - oy, xs - ox]) == k:
                            heatmap_rf[30 * (ys - oy): 30 * (ys - oy + 1), 30 * (xs - ox): 30 * (xs - ox + 1), ] = (
                                    np.array(Image.open(simple_cell.weight_images[0])) * weight_sc
                            )

                        pos_x = xc * (11 * spinet.l_shape[layer, 0] + 10) + (xs - ox) * 11
                        pos_y = (
                                yc
                                * (
                                        11 * spinet.l_shape[layer, 1] * spinet.n_shape[layer, 2]
                                        + spinet.n_shape[layer, 2] * 2
                                        + 10
                                )
                                + z * (11 * spinet.l_shape[layer, 1] + 2)
                                + (ys - oy) * 11
                        )
                        pdf.image(path, x=pos_x, y=pos_y, w=10, h=10)
            plt.figure()
            plt.matshow(heatmap)
            plt.savefig(spinet.path + "figures/1/" + str(c), bbox_inches="tight")
            # h_max = np.max(heatmap.flatten())
            # for i in range(4):
            #     for j in range(4):
            #         heatmap_rf[30*i:30*(i+1), 30*j:30*(j+1)] = heatmap_rf[30*i:30*(i+1), 30*j:30*(j+1)] * heatmap[i, j] / h_max
            Image.fromarray(heatmap_rf.astype("uint8")).save(
                spinet.path + "figures/1/" + str(c) + "_rf.png"
            )
    return pdf


def sort_connections(spinet, cell, depth):
    strengths = []
    for z in range(depth):
        strengths.append(np.sum(cell.weights[:, :, z]))
    return np.argsort(strengths)[::-1]


def load_array_param(spinet, param):
    simple_array = np.zeros(spinet.nb_simple_cells)
    for i, simple_cell in enumerate(spinet.neurons[0]):
        simple_array[i] = simple_cell.params[param]
    return simple_array, 0
    simple_array = simple_array.reshape(
        (len(spinet.l1xanchor) * spinet.l1width, len(spinet.l1yanchor) * spinet.l1height, spinet.l1depth,)
    ).transpose((1, 0, 2))

    complex_array = np.zeros(spinet.nb_complex_cells)
    for i, complex_cell in enumerate(spinet.complex_cells):
        complex_array[i] = complex_cell.params[param]
    complex_array = complex_array.reshape(
        (len(spinet.l2xanchor) * spinet.l2width, len(spinet.l2yanchor) * spinet.l2height, spinet.l2depth,)
    ).transpose((1, 0, 2))

    return simple_array, complex_array


def complex_cells_directions(spinet, rotations):
    dir_vec = plot_directions(spinet, spinet.directions, rotations)
    ori_vec = plot_orientations(spinet, spinet.orientations, rotations)
    return dir_vec, ori_vec


def plot_directions(spinet, directions, rotations):
    temp = np.zeros((directions.shape[0] + 1, directions.shape[1]))
    temp[:-1, :] = directions
    temp[-1, :] = directions[0, :]
    angles = np.append(rotations, 0) * np.pi / 180
    spike_vector = temp

    vectors = create_figure(spinet, spike_vector, angles, "complex_directions")
    return vectors


def plot_orientations(spinet, orientations, rotations):
    temp = np.zeros((orientations.shape[0] + 1, orientations.shape[1]))
    temp[:-1, :] = orientations
    temp[-1, :] = orientations[0, :]
    angles = np.append(rotations[::2], 0) * np.pi / 180
    spike_vector = temp

    vectors = create_figure(spinet, spike_vector, angles, "complex_orientations")
    return vectors


def create_figure(spinet, spike_vector, angles, name):
    vectors = []
    for i in range(spike_vector.shape[1]):
        mean = mean_response(spike_vector[:-1, i], angles[:-1])
        vectors.append(mean)
        plt.figure()
        ax = plt.subplot(111, polar=True)
        # ax.set_title("Cell "+str(i))
        ax.plot(angles, spike_vector[:, i], "darkslategrey")
        ax.arrow(
            np.angle(mean),
            0,
            0,
            2 * np.abs(mean),
            width=0.02,
            head_width=0,
            head_length=0,
            length_includes_head=True,
            edgecolor="firebrick",
            lw=2,
            zorder=5,
        )
        ax.set_thetamax(360)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=13)
        ax.set_xticklabels(["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°"])
        plt.savefig(spinet.path + "figures/" + name + "/" + str(i), bbox_inches="tight")
        return vectors


def mean_response(directions, angles):
    return np.mean(directions * np.exp(1j * angles))


def display_network(spinets):
    for spinet in spinets:
        spinet.generate_weight_images()

        if spinet.conf["sharingType"] == "patch":
            for i in range(spinet.conf["nbCameras"]):
                pdf = pdf_weight_sharing(spinet, i)
                pdf.output(
                    spinet.path + "figures/0/weight_sharing_" + str(i) + ".pdf", "F",
                )
            if spinet.conf["nbCameras"] == 2:
                pdf = pdf_weight_sharing_left_right_combined(spinet)
                pdf.output(
                    spinet.path + "figures/0/weight_sharing_combined.pdf", "F",
                )
        elif spinet.conf["sharingType"] == "none":
            for layer in range(spinet.l_shape[0, 2]):
                for i in range(spinet.conf["nbCameras"]):
                    pdf = pdf_simple_cell(spinet, layer, i)
                    pdf.output(
                        spinet.path + "figures/0/" + str(layer) + "_" + str(i) + ".pdf", "F",
                    )
                pdf = pdf_simple_cell_left_right_combined(spinet, layer)
                pdf.output(
                    spinet.path + "figures/0/" + str(layer) + "_combined.pdf", "F",
                )
            # pdf = generate_pdf_layers(spinet, spinet.l1height, spinet.l1width, spinet.neuron1_synapses, spinet.l1depth)
            # pdf.output(spinet.path+"figures/multi_layer.pdf", "F")

        if len(spinet.neurons[1]) > 0:
            os.mkdir(spinet.path + "figures/1/tmp/")
            # for z in range(spinet.l_shape[1, 2]):
            #     pdfs = pdf_complex_cell(spinet, z, 1)
            shutil.rmtree(spinet.path + "figures/1/tmp/")
