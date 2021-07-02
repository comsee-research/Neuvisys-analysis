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


<<<<<<< HEAD
def clean_network(path, simple_cells, complex_cells, motor_cells, json_only):
=======
def clean_network(path, simple_cells, complex_cells, json_only):
>>>>>>> 1293d93e4fed97b25734913b2d32e474eef1a407
    if json_only:
        if simple_cells:
            for file in os.listdir(path + "weights/simple_cells/"):
                if file.endswith(".json"):
                    os.remove(path + "weights/simple_cells/" + file)
        if complex_cells:
            for file in os.listdir(path + "weights/complex_cells/"):
                if file.endswith(".json"):
                    os.remove(path + "weights/complex_cells/" + file)
<<<<<<< HEAD
        if motor_cells:
            for file in os.listdir(path + "weights/motor_cells/"):
                if file.endswith(".json"):
                    os.remove(path + "weights/motor_cells/" + file)
=======
>>>>>>> 1293d93e4fed97b25734913b2d32e474eef1a407
    else:
        if simple_cells:
            delete_files(path + "weights/simple_cells/")
        if complex_cells:
            delete_files(path + "weights/complex_cells/")
<<<<<<< HEAD
        if motor_cells:
            delete_files(path + "weights/motor_cells/")
=======
>>>>>>> 1293d93e4fed97b25734913b2d32e474eef1a407
        delete_files(path + "images/complex_connections/")
        delete_files(path + "images/simple_cells/")
        delete_files(path + "images/complex_cells/")
        os.remove(path + "learning_trace.txt")

<<<<<<< HEAD

=======
>>>>>>> 1293d93e4fed97b25734913b2d32e474eef1a407
class SpikingNetwork:
    """Spiking Neural Network class"""

    def __init__(self, path):
        self.path = path
        with open(path + "configs/network_config.json") as file:
            self.conf = json.load(file)
        with open(path + "configs/simple_cell_config.json") as file:
            self.simple_conf = json.load(file)
        with open(path + "configs/complex_cell_config.json") as file:
            self.complex_conf = json.load(file)
        with open(path + "configs/motor_cell_config.json") as file:
            self.motor_conf = json.load(file)

        self.nb_neurons = 0
        self.simple_cells, self.sspikes = self.load_weights(
            "weights/simple_cells", "simple_cell"
        )
        self.complex_cells, self.cspikes = self.load_weights(
            "weights/complex_cells", "complex_cell"
        )
        self.motor_cells, self.mspikes = self.load_weights(
            "weights/motor_cells", "motor_cell"
        )

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

    def load_weights(self, cell_path, cell_type):
        neurons = []
        spike_train = []

        neurons_paths = natsorted(os.listdir(self.path + cell_path))
        for paths in [
            neurons_paths[i : i + 2] for i in range(0, len(neurons_paths), 2)
        ]:
            neuron = Neuron(
                cell_type,
                self.path + "configs/" + cell_type + "_config.json",
                self.path + "weights/" + cell_type + "s/",
                *paths
            )
            neurons.append(neuron)
            if neuron.conf["TRACKING"] == "partial":
                spike_train.append(neuron.params["spike_train"])
        self.nb_neurons += len(neurons)
        return neurons, spike_train

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

    def save_complex_directions(self, spikes, rotations):
        spike_vector = []
        for rot in range(rotations.size):
            spike_vector.append(np.count_nonzero(spikes[rot], axis=1))
        spike_vector = np.array(spike_vector)

        np.save(self.path + "gabors/data/direction_response", spike_vector)
        self.directions = spike_vector
        self.orientations = self.directions[0:8] + self.directions[8:16]
        
    def spike_rate(self):
        time = np.max(self.sspikes)
        srates = np.count_nonzero(self.sspikes, axis=1) / (time * 1e-6)
        return np.mean(srates), np.std(srates)

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
