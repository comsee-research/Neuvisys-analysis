#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:32:36 2020

@author: thomas
"""

from natsort import natsorted
import json
import os
from PIL import Image
import scipy.io as sio
import numpy as np
import itertools
from aedat_tools.aedat_tools import delete_files


def compress_weight(weights, path):
    img = np.array(255 * (weights / weights.max()), dtype=np.uint8)
    img = Image.fromarray(img).save(path)


def reshape_weights(weights, width, height):
    weights = np.concatenate((weights, np.zeros((1, width, height))), axis=0)
    return np.kron(np.swapaxes(weights, 0, 2), np.ones((3, 3, 1)))


class SpikingNetwork:
    """Spiking Neural Network class"""

    def __init__(self, path):
        self.path = path
        with open(path + "configs/network_config.json") as file:
            self.conf = json.load(file)
        self.neurons = []
        self.simple_cells = []
        self.complex_cells = []

        self.sspikes = []
        self.cspikes = []

        neurons_paths = natsorted(os.listdir(path + "weights/complex_cells"))
        for paths in [
            neurons_paths[i : i + 2] for i in range(0, len(neurons_paths), 2)
        ]:
            neuron = Neuron(
                "complex",
                path + "configs/complex_cell_config.json",
                path + "weights/complex_cells/",
                *paths
            )
            self.neurons.append(neuron)
            self.complex_cells.append(neuron)
            if neuron.conf["TRACKING"] == "partial":
                self.cspikes.append(neuron.params["spike_train"])

        neurons_paths = natsorted(os.listdir(path + "weights/simple_cells"))
        for paths in [
            neurons_paths[i : i + 2] for i in range(0, len(neurons_paths), 2)
        ]:
            neuron = Neuron(
                "simple",
                path + "configs/simple_cell_config.json",
                path + "weights/simple_cells/",
                *paths
            )
            self.neurons.append(neuron)
            self.simple_cells.append(neuron)

            if neuron.conf["TRACKING"] == "partial":
                self.sspikes.append(neuron.params["spike_train"])

        self.nb_neurons = len(self.neurons)
        self.nb_simple_cells = len(self.simple_cells)
        self.nb_complex_cells = len(self.complex_cells)

        self.layout1 = np.load(path + "weights/layout1.npy")
        # self.layout2 = np.load(path + "weights/layout2.npy")

        if np.array(self.sspikes, dtype=object).size > 0:
            self.sspikes = np.array(
                list(itertools.zip_longest(*self.sspikes, fillvalue=0))
            ).T
            self.sspikes[self.sspikes != 0] -= np.min(self.sspikes[self.sspikes != 0])
        if np.array(self.cspikes, dtype=object).size > 0:
            self.cspikes = np.array(
                list(itertools.zip_longest(*self.cspikes, fillvalue=0))
            ).T
            self.cspikes[self.cspikes != 0] -= np.min(self.cspikes[self.cspikes != 0])

        if os.path.exists(self.path + "gabors/data/direction_response.npy"):
            self.directions = np.load(self.path + "gabors/data/direction_response.npy")
            self.orientations = self.directions[0:8] + self.directions[8:16]

    def generate_weight_images(self):
        for i, neuron in enumerate(self.simple_cells):
            for synapse in range(self.conf["Neuron1Synapses"]):
                for camera in range(self.conf["NbCameras"]):
                    weights = reshape_weights(
                        neuron.weights[:, camera, synapse],
                        self.conf["Neuron1Width"],
                        self.conf["Neuron1Height"],
                    )
                    path = (
                        self.path
                        + "images/simple_cells/"
                        + str(i)
                        + "_syn"
                        + str(synapse)
                        + "_cam"
                        + str(camera)
                        + ".png"
                    )
                    compress_weight(weights, path)
                    neuron.weight_images.append(path)

        for i, neuron in enumerate(self.complex_cells):
            for lay in range(self.conf["Neuron2Depth"]):
                dim = np.zeros((self.conf["Neuron2Width"], self.conf["Neuron2Height"]))
                weight = np.stack((neuron.weights[:, :, lay], dim, dim), axis=2)
                path = (
                    self.path
                    + "images/complex_cells/"
                    + str(i)
                    + "_lay_"
                    + str(lay)
                    + ".png"
                )
                compress_weight(np.kron(weight, np.ones((7, 7, 1))), path)
                neuron.weight_images.append(path)

    def get_weights(self, neuron_type):
        if neuron_type == "simple":
            weights = []
            if self.conf["SharingType"] == "full":
                weights = [
                    neuron.weights
                    for neuron in self.simple_cells[0 : self.conf["L1Depth"]]
                ]
            elif self.conf["SharingType"] == "patch":
                for i in range(
                    0,
                    self.nb_simple_cells,
                    self.conf["L1Depth"] * self.conf["L1Width"] * self.conf["L1Height"],
                ):
                    weights += [
                        neuron.weights
                        for neuron in self.simple_cells[i : i + self.conf["L1Depth"]]
                    ]
            else:
                weights = [neuron.weights for neuron in self.simple_cells]
            return np.array(weights)

    def generate_weight_mat(self):
        weights = self.get_weights("simple")

        w = self.conf["Neuron1Width"] * self.conf["Neuron1Height"]
        basis = np.zeros((2 * w, len(weights)))
        for c in range(self.conf["NbCameras"]):
            for i, weight in enumerate(weights):
                basis[c * w : (c + 1) * w, i] = (
                    weight[0, c, 0] - weight[1, c, 0]
                ).flatten("F")
        sio.savemat(self.path + "gabors/data/weights.mat", {"basis": basis})

        return basis

    def clean_network(self, simple_cells, complex_cells, json_only):
        if json_only:
            if simple_cells:
                for file in os.listdir(self.path + "weights/simple_cells/"):
                    if file.endswith(".json"):
                        os.remove(self.path + "weights/simple_cells/" + file)
            if complex_cells:
                for file in os.listdir(self.path + "weights/complex_cells/"):
                    if file.endswith(".json"):
                        os.remove(self.path + "weights/complex_cells/" + file)
        else:
            if simple_cells:
                delete_files(self.path + "weights/simple_cells/")
            if complex_cells:
                delete_files(self.path + "weights/complex_cells/")
            delete_files(self.path + "images/complex_connections/")
            delete_files(self.path + "images/simple_cells/")
            delete_files(self.path + "images/complex_cells/")
            os.remove(self.path + "learning_trace.txt")

    def save_complex_directions(self, spikes, rotations):
        spike_vector = []
        for rot in range(rotations.size):
            spike_vector.append(np.count_nonzero(spikes[rot], axis=1))
        spike_vector = np.array(spike_vector)

        np.save(self.path + "gabors/data/direction_response", spike_vector)
        self.directions = spike_vector
        self.orientations = self.directions[0:8] + self.directions[8:16]


class Neuron:
    """Spiking Neuron class"""

    def __init__(self, neuron_type, config_path, path, param_path, weight_path):
        self.type = neuron_type
        with open(config_path) as file:
            self.conf = json.load(file)
        with open(path + param_path) as file:
            self.params = json.load(file)
        self.weights = np.load(path + weight_path)
        self.weight_images = []
        self.gabor_image = 0
        self.lambd = 0
        self.theta = 0
        self.phase = 0
        self.sigma = 0
        self.error = 0

    def add_gabor(self, image, mu, sigma, lambd, phase, theta, error):
        self.gabor_image = image
        self.mu = mu
        self.sigma = sigma
        self.lambd = lambd
        self.phase = phase
        self.theta = theta
        self.error = error
        self.orientation = self.theta
