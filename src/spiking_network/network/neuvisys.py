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
from src.events.tools.read_write.events_tools import delete_files


def compress_weight(weights, path):
    img = np.array(255 * (weights / weights.max()), dtype=np.uint8)
    img = Image.fromarray(img).save(path)


def reshape_weights(weights, width, height):
    weights = np.concatenate((weights, np.zeros((1, width, height))), axis=0)
    return np.kron(np.swapaxes(weights, 0, 2), np.ones((3, 3, 1)))


def clean_network(path, layers):
    for i in layers:
        delete_files(path + "weights/" + str(i) + "/")
        delete_files(path + "images/" + str(i) + "/")
    os.remove(path + "networkState.json")


class SpikingNetwork:
    """Spiking Neural Network class"""

    def __init__(self, path):
        self.path = path
        try:
            with open(path + "configs/network_config.json") as file:
                self.conf = json.load(file)
            with open(path + "configs/simple_cell_config.json") as file:
                self.simple_conf = json.load(file)
            with open(path + "configs/complex_cell_config.json") as file:
                self.complex_conf = json.load(file)
            with open(path + "configs/critic_cell_config.json") as file:
                self.critic_conf = json.load(file)
            with open(path + "configs/actor_cell_config.json") as file:
                self.actor_conf = json.load(file)
            with open(path + "networkState.json") as file:
                self.state = json.load(file)
        except:
            print("could not load some config file")

        self.nb_neurons = 0
        self.neurons = []
        self.spikes = []
        layer1, spikes1 = self.load_weights(0, "simple_cell", "simple_cell_config.json")
        self.neurons.append(layer1)
        self.spikes.append(spikes1)

        layer2, spikes2 = self.load_weights(1, "complex_cell", "complex_cell_config.json")
        self.neurons.append(layer2)
        self.spikes.append(spikes2)

        layer3, spikes3 = self.load_weights(2, "motor_cell", "critic_cell_config.json")
        self.neurons.append(layer3)
        self.spikes.append(spikes3)

        layer4, spikes4 = self.load_weights(3, "motor_cell", "actor_cell_config.json")
        self.neurons.append(layer4)
        self.spikes.append(spikes4)

        self.layout = []
        self.layout.append(np.load(path + "weights/layout_0.npy"))
        self.layout.append(np.load(path + "weights/layout_1.npy"))
        self.layout.append(np.load(path + "weights/layout_2.npy"))
        self.layout.append(np.load(path + "weights/layout_3.npy"))

        for i in range(len(self.spikes)):
            if np.array(self.spikes[i], dtype=object).size > 0:
                self.spikes[i] = np.array(list(itertools.zip_longest(*self.spikes[i], fillvalue=0))).T
                self.spikes[i][self.spikes[i] != 0] -= np.min(self.spikes[i][self.spikes[i] != 0])

        if os.path.exists(self.path + "gabors/data/direction_response.npy"):
            self.directions = np.load(self.path + "gabors/data/direction_response.npy")
            self.orientations = self.directions[0:8] + self.directions[8:16]

        self.p_shape = np.array(self.conf["layerPatches"])
        self.l_shape = np.array(self.conf["layerSizes"])
        self.n_shape = np.array(self.conf["neuronSizes"])

    def load_weights(self, layer, cell_type, config):
        neurons = []
        spike_train = []

        neurons_paths = natsorted(os.listdir(self.path + "weights/" + str(layer) + "/"))
        for paths in [neurons_paths[i: i + 2] for i in range(0, len(neurons_paths), 2)]:
            neuron = Neuron(
                cell_type,
                self.path + "configs/" + config,
                self.path + "weights/" + str(layer) + "/",
                *paths
            )
            neurons.append(neuron)
            if neuron.conf["TRACKING"] == "partial":
                spike_train.append(neuron.params["spike_train"])
        self.nb_neurons += len(neurons)
        return neurons, spike_train

    def generate_weight_images(self):
        for layer in range(self.p_shape.shape[0]):
            if layer == 0:
                for i, neuron in enumerate(self.neurons[layer]):
                    for synapse in range(self.conf["Neuron1Synapses"]):
                        for camera in range(self.conf["NbCameras"]):
                            weights = reshape_weights(
                                neuron.weights[:, camera, synapse], self.n_shape[layer, 0], self.n_shape[layer, 1],
                            )
                            path = (
                                    self.path
                                    + "images/0/"
                                    + str(i)
                                    + "_syn"
                                    + str(synapse)
                                    + "_cam"
                                    + str(camera)
                                    + ".png"
                            )
                            compress_weight(weights, path)
                            neuron.weight_images.append(path)
            else:
                for i, neuron in enumerate(self.neurons[layer]):
                    for z in range(self.n_shape[layer, 2]):
                        dim = np.zeros((self.n_shape[layer, 0], self.n_shape[layer, 1]))
                        weight = np.stack((neuron.weights[:, :, z], dim, dim), axis=2)
                        path = self.path + "images/" + str(layer) + "/" + str(i) + "_lay_" + str(z) + ".png"
                        compress_weight(np.kron(weight, np.ones((7, 7, 1))), path)
                        neuron.weight_images.append(path)

    def get_weights(self, neuron_type):
        if neuron_type == "simple":
            weights = []
            if self.conf["SharingType"] == "full":
                weights = [neuron.weights for neuron in self.simple_cells[0: self.conf["L1Depth"]]]
            elif self.conf["SharingType"] == "patch":
                for i in range(
                        0, self.nb_simple_cells, self.conf["L1Depth"] * self.conf["L1Width"] * self.conf["L1Height"],
                ):
                    weights += [neuron.weights for neuron in self.simple_cells[i: i + self.conf["L1Depth"]]]
            else:
                weights = [neuron.weights for neuron in self.simple_cells]
            return np.array(weights)

    def generate_weight_mat(self):
        weights = self.get_weights("simple")

        w = self.conf["Neuron1Width"] * self.conf["Neuron1Height"]
        basis = np.zeros((2 * w, len(weights)))
        for c in range(self.conf["NbCameras"]):
            for i, weight in enumerate(weights):
                basis[c * w: (c + 1) * w, i] = (weight[0, c, 0] - weight[1, c, 0]).flatten("F")
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
