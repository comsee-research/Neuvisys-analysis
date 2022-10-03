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
from fpdf import FPDF
from natsort import natsorted
from PIL import Image


def pdf_simple_cell(spinet, layer, camera):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[0, 0], 11 * spinet.l_shape[0, 1] * spinet.conf["neuron1Synapses"],),
    )
    pdf.add_page()

    for neuron in spinet.neurons[0]:
        x, y, z = neuron.params["position"]
        if z == layer:
            for i in range(spinet.conf["neuron1Synapses"]):
                pos_x = x * 11
                pos_y = y * spinet.conf["neuron1Synapses"] * 11 + i * 11
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_complex_cell(spinet, zcell):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[1, 0], 11 * spinet.l_shape[1, 1],),
    )
    pdf.add_page()

    for neuron in spinet.neurons[1]:
        x, y, z = neuron.params["position"]
        if z == zcell:
            pos_x = x * 11
            pos_y = y * 11
            pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_simple_cell_left_right_combined(spinet, layer):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[0, 0], 24 * spinet.l_shape[0, 1] * spinet.conf["neuron1Synapses"],),
    )
    pdf.add_page()

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


def pdf_weight_sharing(spinet, nb_cameras, camera):
    side = int(np.sqrt(spinet.l_shape[0, 2]))
    if nb_cameras == 1:
        pad = 11
    else:
        pad = 24
    xpatch = len(spinet.p_shape[0, 0])
    ypatch = len(spinet.p_shape[0, 1])
    pdf = FPDF("P", "mm", (11 * xpatch * side + (xpatch - 1) * 10, pad * ypatch * side + (ypatch - 1) * 10))
    pdf.add_page()

    shift = np.arange(spinet.l_shape[0, 2]).reshape((side, side))
    cell_range = range(0, len(spinet.neurons[0]), spinet.l_shape[0, 2] * spinet.l_shape[0, 0] * spinet.l_shape[0, 1])
    for i in cell_range:
        for neuron in spinet.neurons[0][i: i + spinet.l_shape[0, 2]]:
            x, y, z = neuron.params["position"]
            pos_x = (
                    (x // spinet.l_shape[0, 0]) * side * 11
                    + np.where(shift == z)[0][0] * 11
                    + (x // spinet.l_shape[0, 0]) * 10
            )  # patch size + weight sharing shift + patch padding
            pos_y = (
                    (y // spinet.l_shape[0, 1]) * side * pad
                    + np.where(shift == z)[1][0] * pad
                    + (y // spinet.l_shape[0, 1]) * 10
            )
            if nb_cameras == 1:
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
            else:
                pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
                pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def pdf_weight_sharing_full(spinet, nb_cameras, camera):
    side = int(np.sqrt(spinet.l_shape[0, 2]))
    if nb_cameras == 1:
        pad = 11
    else:
        pad = 24
    pdf = FPDF("P", "mm", (11 * side, pad * side))
    pdf.add_page()

    shift = np.arange(spinet.l_shape[0, 2]).reshape((side, side))
    for neuron in spinet.neurons[0][0:spinet.l_shape[0, 2]]:
        x, y, z = neuron.params["position"]
        pos_x = (
                (x // spinet.l_shape[0, 0]) * side * 11
                + np.where(shift == z)[0][0] * 11
        )  # patch size + weight sharing shift
        pos_y = (
                (y // spinet.l_shape[0, 1]) * side * pad
                + np.where(shift == z)[1][0] * pad
        )
        if nb_cameras == 1:
            pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
        else:
            pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
            pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def display_motor_cell(spinet):
    for i, motor_cell in enumerate(spinet.motor_cells):
        heatmap = np.mean(motor_cell.weights, axis=2)

        plt.figure()
        plt.matshow(heatmap)
        plt.savefig(spinet.path + "figures/motor_figures/" + str(i), bbox_inches="tight")


def pdf_complex_receptive_fields(spinet, layer):
    for c, complex_cell in enumerate(spinet.neurons[layer]):
        ox, oy, oz = complex_cell.params["offset"]

        heatmap = np.zeros((spinet.n_shape[layer, 0, 0], spinet.n_shape[layer, 0, 1]))
        heatmap_rf = np.zeros((120, 120, 3))

        maximum = np.max(complex_cell.weights)
        for i in range(ox, ox + spinet.n_shape[1, 0, 0]):
            for j in range(oy, oy + spinet.n_shape[1, 0, 1]):
                for k in range(spinet.n_shape[layer, 0, 2]):
                    simple_cell = spinet.neurons[0][spinet.layout[0][i, j, k]]
                    xs, ys, zs = simple_cell.params["position"]

                    weight_sc = complex_cell.weights[xs - ox, ys - oy, k] / maximum
                    heatmap[ys - oy, xs - ox] += weight_sc
                    if np.argmax(complex_cell.weights[xs - ox, ys - oy]) == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        heatmap_rf[30 * (ys - oy): 30 * (ys - oy + 1), 30 * (xs - ox): 30 * (xs - ox + 1)] = np.array(
                            sc_weight_image) * weight_sc
        # fig = plt.figure()
        # plt.matshow(heatmap)
        # plt.savefig(spinet.path + "figures/1/" + str(c), bbox_inches="tight")
        # plt.close(fig)
        Image.fromarray(heatmap_rf.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf.png"
        )


def pdf_complex_to_simple_cell_orientations(spinet, zcell, layer):
    pdf = FPDF(
        "P",
        "mm",
        (
            spinet.p_shape[0, 0].shape[0] * spinet.l_shape[0, 0] * 11 + spinet.p_shape[0, 0].shape[0] * 11,
            spinet.p_shape[0, 1].shape[0] * spinet.l_shape[0, 1] * spinet.n_shape[layer, 0, 2] * 11
            + spinet.p_shape[0, 1].shape[0] * 11
            + spinet.p_shape[0, 1].shape[0] * 10,
        ),
    )
    pdf.add_page()

    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "")

    for c, complex_cell in enumerate(spinet.neurons[layer]):
        xc, yc, zc = complex_cell.params["position"]
        ox, oy, oz = complex_cell.params["offset"]

        if zc == zcell:
            maximum = np.max(complex_cell.weights)
            for ind, k in enumerate(sort_connections(complex_cell, spinet.n_shape[layer, 0, 2])):
                for i in range(ox, ox + 4):
                    for j in range(oy, oy + 4):
                        simple_cell = spinet.neurons[0][spinet.layout[0][i, j, k]]
                        xs, ys, zs = simple_cell.params["position"]

                        weight_sc = complex_cell.weights[xs - ox, ys - oy, k] / maximum
                        img = weight_sc * np.array(Image.open(simple_cell.weight_images[0]))
                        path = (spinet.path
                                + "figures/1/tmp/"
                                + str(c)
                                + "_simple_"
                                + str(spinet.layout[layer - 1][i, j, k])
                                + ".png")
                        Image.fromarray(img.astype("uint8")).save(path)

                        pos_x = xc * (11 * spinet.l_shape[0, 0] + 10) + (xs - ox) * 11
                        pos_y = (yc * (11 * spinet.l_shape[0, 1] * spinet.n_shape[layer, 0, 2]
                                       + spinet.n_shape[layer, 0, 2] * 2 + 10)
                                 + ind * (11 * spinet.l_shape[0, 1] + 2) + (ys - oy) * 11)
                        pdf.image(path, x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def sort_connections(cell, depth):
    strengths = []
    for z in range(depth):
        strengths.append(np.sum(cell.weights[:, :, z]))
    return np.argsort(strengths)[::-1]


def load_array_param(spinet, param):
    simple_array = np.zeros(len(spinet.neurons[0]))
    for i, simple_cell in enumerate(spinet.neurons[0]):
        simple_array[i] = simple_cell.params[param]
    return simple_array, 0
    # simple_array = simple_array.reshape(
    #     (len(spinet.l1xanchor) * spinet.l1width, len(spinet.l1yanchor) * spinet.l1height, spinet.l1depth,)
    # ).transpose((1, 0, 2))
    #
    # complex_array = np.zeros(spinet.nb_complex_cells)
    # for i, complex_cell in enumerate(spinet.complex_cells):
    #     complex_array[i] = complex_cell.params[param]
    # complex_array = complex_array.reshape(
    #     (len(spinet.l2xanchor) * spinet.l2width, len(spinet.l2yanchor) * spinet.l2height, spinet.l2depth,)
    # ).transpose((1, 0, 2))
    #
    # return simple_array, complex_array


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

    vectors = create_figures(spinet, spike_vector, angles, "complex_directions")
    return vectors


def plot_orientations(spinet, orientations, rotations):
    temp = np.zeros((orientations.shape[0] + 1, orientations.shape[1]))
    temp[:-1, :] = orientations
    temp[-1, :] = orientations[0, :]
    angles = np.append(rotations[::2], 0) * np.pi / 180
    spike_vector = temp

    vectors = create_figures(spinet, spike_vector, angles, "complex_orientations")
    return vectors


def complex_cell_disparities(spinet, disparities, disp):
    ndisp = disp + np.abs(np.min(disp))
    ndisp = ndisp / np.max(ndisp) * 8

    for i, disparity in enumerate(disparities.T):
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.set_title("Cell " + str(i))
        print(disparity)
        plt.bar(ndisp, disparity)

        ax.set_thetamax(360)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=13)
        ax.set_xticklabels(disp[::2])
        if not os.path.exists(spinet.path + "figures/1/disparities/complex_disparities"):
            os.mkdir(spinet.path + "figures/1/disparities/complex_disparities")
        plt.savefig(spinet.path + "figures/1/disparities/complex_disparities/" + str(i), bbox_inches="tight")
        plt.show()


def create_figures(spinet, spike_vector, angles, name):
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
                pdf = pdf_weight_sharing(spinet, 1, i)
                pdf.output(spinet.path + "figures/0/weight_sharing_" + str(i) + ".pdf", "F")
            if spinet.conf["nbCameras"] == 2:
                pdf = pdf_weight_sharing(spinet, spinet.conf["nbCameras"], 0)
                pdf.output(spinet.path + "figures/0/weight_sharing_combined.pdf", "F")
        elif spinet.conf["sharingType"] == "full":
            for i in range(spinet.conf["nbCameras"]):
                pdf = pdf_weight_sharing_full(spinet, 1, i)
                pdf.output(spinet.path + "figures/0/weight_sharing_" + str(i) + ".pdf", "F")
            if spinet.conf["nbCameras"] == 2:
                pdf = pdf_weight_sharing_full(spinet, spinet.conf["nbCameras"], 0)
                pdf.output(spinet.path + "figures/0/weight_sharing_combined.pdf", "F")
        elif spinet.conf["sharingType"] == "none":
            for layer in range(spinet.l_shape[0, 2]):
                for i in range(spinet.conf["nbCameras"]):
                    pdf = pdf_simple_cell(spinet, layer, i)
                    pdf.output(spinet.path + "figures/0/" + str(layer) + "_" + str(i) + ".pdf", "F")
                pdf = pdf_simple_cell_left_right_combined(spinet, layer)
                pdf.output(spinet.path + "figures/0/" + str(layer) + "_combined.pdf", "F")

        if len(spinet.neurons) > 1:
            pdf_complex_receptive_fields(spinet, 1)
            for z in range(spinet.l_shape[1, 2]):
                pdf = pdf_complex_cell(spinet, z)
                pdf.output(spinet.path + "figures/1/complex_weights_depth_" + str(z) + ".pdf", "F")
