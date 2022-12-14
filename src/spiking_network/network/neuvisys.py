#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:32:36 2020

@author: thomas
"""

import itertools
import json
import os
import random
import re
import shutil

import numpy as np
from PIL import Image
from natsort import natsorted


def delete_files(folder):
    for file in os.scandir(folder):
        try:
            if os.path.isfile(file.path) or os.path.islink(file.path):
                os.unlink(file.path)
            elif os.path.isdir(file.path):
                shutil.rmtree(file.path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file.path, e))


def compress_weight(weights, path, max_weight):
    img = np.array(255 * (weights / max_weight), dtype=np.uint8)
    Image.fromarray(img).save(path)


def reshape_weights(weights, width, height):
    weights = np.concatenate((weights, np.zeros((1, width, height))), axis=0)
    return np.kron(np.swapaxes(weights, 0, 2), np.ones((3, 3, 1)))


def shuffle_weights(path):
    neurons_paths = natsorted(os.listdir(path))
    for pattern in [".*tdi", ".*li"]:
        weight_files = list(filter(re.compile(pattern).match, neurons_paths))
        shuffled_weight_files = random.sample(weight_files, len(weight_files))

        for old_name, new_name in zip(weight_files, shuffled_weight_files):
            os.rename(path + old_name, path + new_name + "bis")

        for name in weight_files:
            os.rename(path + name + "bis", path + name)


def clean_network(path, layers):
    for i in layers:
        delete_files(path + "weights/" + str(i) + "/")
        delete_files(path + "images/" + str(i) + "/")
    os.remove(path + "networkState.json")


class SpikingNetwork:
    """Spiking Neural Network class"""

    def __init__(self, path, loading=None):
        self.path = path
        try:
            with open(path + "configs/network_config.json") as file:
                self.conf = json.load(file)
            with open(path + "configs/rl_config.json") as file:
                self.rl_conf = json.load(file)
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
        except FileNotFoundError as e:
            print(e)

        self.nb_neurons = 0
        self.neurons = []
        self.weights = []
        self.spikes = []
        self.layout = []
        self.shared_id = []
        type_to_config = {"SimpleCell": "simple_cell_config.json", "ComplexCell": "complex_cell_config.json",
                          "CriticCell": "critic_cell_config.json", "ActorCell": "actor_cell_config.json"}

        self.p_shape = np.array(self.conf["layerPatches"], dtype=object)
        self.l_shape = np.array(self.conf["layerSizes"])
        self.n_shape = np.array(self.conf["neuronSizes"], dtype=object)

        for layer, neuron_type in enumerate(self.conf["neuronType"]):
            if loading is not None:
                if loading[layer]:
                    neurons, spikes = self.load_neurons(layer, neuron_type, type_to_config[neuron_type])
                else:
                    neurons = []
                    spikes = []
                self.neurons.append(neurons)
                self.spikes.append(spikes)
                self.layout.append(np.load(path + "weights/layout_" + str(layer) + ".npy"))
                if loading[layer]:
                    self.weights.append(self.load_weights(layer, neuron_type))

        for i in range(len(self.spikes)):
            if np.array(self.spikes[i], dtype=object).size > 0:
                self.spikes[i] = np.array(list(itertools.zip_longest(*self.spikes[i], fillvalue=0))).T
                # self.spikes[i][self.spikes[i] != 0] -= np.min(self.spikes[i][self.spikes[i] != 0])

        if os.path.exists(self.path + "gabors/0/rotation_response.npy"):
            self.directions = []
            self.orientations = []
            for layer, neuron_type in enumerate(self.conf["neuronType"]):
                if layer < 2:
                    self.directions.append(np.load(self.path + "gabors/" + str(layer) + "/rotation_response.npy"))
                    self.orientations.append(self.directions[layer][0:8] + self.directions[layer][8:16])

        if os.path.exists(self.path + "gabors/data/disparity_response.npy"):
            self.disparities = np.load(self.path + "gabors/data/disparity_response.npy")

    def load_neurons(self, layer, neuron_type, config):
        neurons = []
        spike_train = []

        weight_path = self.path + "weights/" + str(layer) + "/"
        neurons_paths = natsorted(os.listdir(weight_path))
        config_files = list(filter(re.compile(".*json").match, neurons_paths))

        with open(self.path + "configs/" + config) as file:
            config = json.load(file)

        for index in range(len(config_files)):
            with open(self.path + "weights/" + str(layer) + "/" + str(index) + ".json") as file:
                params = json.load(file)
            neuron = Neuron(neuron_type, index, config, params)
            if os.path.exists(weight_path + str(neuron.id) + "li.npy"):
                neuron.link_weights_li(np.load(weight_path + str(neuron.id) + "li.npy"))
            if os.path.exists(weight_path + str(neuron.id) + "tdi.npy"):
                neuron.link_weights_tdi(np.load(weight_path + str(neuron.id) + "tdi.npy"))
            neurons.append(neuron)
            if neuron.conf["TRACKING"] == "partial":
                spike_train.append(neuron.params["spike_train"])
        self.nb_neurons += len(neurons)
        return neurons, spike_train

    def load_weights(self, layer, neuron_type):
        weights = []
        if neuron_type == "SimpleCell" and self.conf["sharingType"] == "patch":
            step = self.l_shape[layer, 0] * self.l_shape[layer, 1] * self.l_shape[layer, 2]
            for r_id in range(0, len(self.neurons[layer]), step):
                for i, neuron in enumerate(self.neurons[layer][r_id:r_id + self.l_shape[layer, 2]]):
                    weights.append(np.load(self.path + "weights/0/" + str(neuron.id) + ".npy"))
                    self.shared_id.append(np.arange(r_id + i, r_id + i + step, self.l_shape[layer, 2]))
            self.shared_id = np.array(self.shared_id)

            for i, weight in enumerate(weights):
                for shared in self.shared_id[i]:
                    self.neurons[layer][shared].link_weights(weight)
        elif neuron_type == "SimpleCell" and self.conf["sharingType"] == "full":
            step = self.l_shape[layer, 0] * self.l_shape[layer, 1] * self.l_shape[layer, 2]
            for i, neuron in enumerate(self.neurons[layer][:self.l_shape[layer, 2]]):
                weights.append(np.load(self.path + "weights/0/" + str(neuron.id) + ".npy"))
                self.shared_id.append(np.arange(i, i + step, self.l_shape[layer, 2]))

            for i, weight in enumerate(weights):
                for shared in self.shared_id[i]:
                    self.neurons[layer][shared].link_weights(weight)
        elif neuron_type == "CriticCell" or neuron_type == "ActorCell":
            for neuron in self.neurons[layer]:
                neuron.link_weights(np.load(self.path + "weights/" + str(layer) + "/" + str(neuron.id) + "_0.npy"))
                weights.append(neuron.weights)
        else:
            for neuron in self.neurons[layer]:
                neuron.link_weights(np.load(self.path + "weights/" + str(layer) + "/" + str(neuron.id) + ".npy"))
                weights.append(neuron.weights)

        return np.array(weights)

    def generate_weight_images(self):
        for layer in range(self.p_shape.shape[0]):
            if layer == 0:
                for i, weights in enumerate(self.weights[layer]):
                    max_weight = np.max(weights)
                    for synapse in range(self.conf["neuron1Synapses"]):
                        for camera in range(self.conf["nbCameras"]):
                            n_weight = reshape_weights(
                                weights[:, camera, synapse], self.n_shape[layer, 0, 0], self.n_shape[layer, 0, 1],
                            )
                            path = (self.path + "images/0/" + str(i) + "_syn" + str(synapse) + "_cam" + str(
                                camera) + ".png")

                            compress_weight(n_weight, path, max_weight)
                            if np.any(self.shared_id):
                                for shared in self.shared_id[i]:
                                    self.neurons[layer][shared].weight_images.append(path)
                            else:
                                self.neurons[layer][i].weight_images.append(path)
            else:
                for i, neuron in enumerate(self.neurons[layer]):
                    weights = np.mean(neuron.weights, axis=2)
                    weights = np.swapaxes(weights, 0, 1)
                    weights = np.stack((weights, np.zeros(weights.shape), np.zeros(weights.shape)), axis=2)
                    path = self.path + "images/" + str(layer) + "/" + str(i) + ".png"
                    compress_weight(np.kron(weights, np.ones((7, 7, 1))), path, weights.max())
                    neuron.weight_images.append(path)
                    # for z in range(self.n_shape[layer, 2]):
                    #     dim = np.zeros((self.n_shape[layer, 0], self.n_shape[layer, 1]))
                    #     weight = np.stack((neuron.weights[:, :, z], dim, dim), axis=2)
                    #     path = self.path + "images/" + str(layer) + "/" + str(i) + "_lay_" + str(z) + ".png"
                    #     compress_weight(np.kron(weight, np.ones((7, 7, 1))), path)
                    #     neuron.weight_images.append(path)

    def generate_weight_mat(self):
        w = self.n_shape[0, 0] * self.n_shape[0, 1]
        basis = np.zeros((2 * w, len(self.weights[0])))
        for c in range(self.conf["nbCameras"]):
            for i, weight in enumerate(self.weights[0]):
                basis[c * w: (c + 1) * w, i] = (weight[0, c, 0] - weight[1, c, 0]).flatten("F")
        # sio.savemat(self.path + "gabors/data/weights.mat", {"basis": basis})

        return basis

    def save_rotation_response(self, spikes, rotations):
        self.directions = []
        self.orientations = []
        for layer, response in enumerate(spikes):
            spike_vector = []
            for rot in range(rotations.size):
                spike_vector.append(np.count_nonzero(response[rot], axis=1))
            spike_vector = np.array(spike_vector)
            np.save(self.path + "gabors/" + str(layer) + "/rotation_response", spike_vector)
            self.directions = spike_vector
            self.orientations = self.directions[0:8] + self.directions[8:16]

    def save_complex_disparities(self, spikes, disparities):
        spike_vector = []
        for disp in range(disparities.size):
            spike_vector.append(np.count_nonzero(spikes[disp], axis=1))
        spike_vector = np.array(spike_vector)

        np.save(self.path + "gabors/data/disparity_response", spike_vector)
        self.disparities = spike_vector

    def spike_rate(self):
        time = np.max(self.spikes)
        srates = np.count_nonzero(self.spikes, axis=1) / (time * 1e-6)
        return np.mean(srates), np.std(srates)


class Neuron:
    """Spiking Neuron class"""

    def __init__(self, neuron_type, index, config, params):
        self.type = neuron_type
        self.id = index
        self.conf = config
        self.params = params
        self.spike_train = np.array(self.params["spike_train"])
        self.weights = 0
        self.weights_li = 0
        self.weights_tdi = 0
        self.weight_images = []
        self.gabor_image = 0
        self.lambd = 0
        self.theta = 0
        self.phase = 0
        self.sigma = 0
        self.error = 0
        self.mu = None
        self.orientation = None
        self.disparity = 0

    def link_weights(self, weights):
        self.weights = weights

    def link_weights_li(self, weights_li):
        self.weights_li = weights_li

    def link_weights_tdi(self, weights_tdi):
        self.weights_tdi = weights_tdi

    def add_gabor(self, image, mu, sigma, lambd, phase, theta, error):
        self.gabor_image = image
        self.mu = mu
        self.sigma = sigma
        self.lambd = lambd
        self.phase = phase
        self.theta = theta
        self.error = error
        self.orientation = self.theta

    def add_disparity(self, disparity):
        self.disparity = disparity

# def load_weights_layer(self, layer, neuron_type):
#     weight_path = self.path + "weights/" + str(layer) + "/weights.npz"
#     weight_li_path = self.path + "weights/" + str(layer) + "/weightsLI.npz"
#     weight_tdi_path = self.path + "weights/" + str(layer) + "/weightsTDI.npz"
#
#     weights = np.load(weight_path)
#     weights_li = 0
#     weights_tdi = 0
#     if os.path.exists(weight_li_path):
#         weights_li = np.load(weight_li_path)
#     if os.path.exists(weight_tdi_path):
#         weights_tdi = np.load(weight_tdi_path)
#
#     patch = False
#     if neuron_type == "SimpleCell" and self.conf["sharingType"] == "patch":
#         patch = True
#         step = self.l_shape[layer, 0] * self.l_shape[layer, 1] * self.l_shape[layer, 2]
#         for r_id in range(0, len(self.neurons[layer]), step):
#             for i, neuron in enumerate(self.neurons[layer][r_id: r_id + self.l_shape[layer, 2]]):
#                 self.shared_id.append(np.arange(r_id + i, r_id + i + step, self.l_shape[layer, 2]))
#         self.shared_id = np.array(self.shared_id)
#
#         for ind in weights:
#             if isinstance(ind, int):
#                 for shared in self.shared_id[ind]:
#                     self.neurons[layer][shared].link_weights(weights[ind])
#
#     for neuron in self.neurons[layer]:
#         if not patch:
#             neuron.link_weights(weights[str(neuron.id)])
#         if os.path.exists(weight_li_path):
#             neuron.link_weights_li(weights_li[str(neuron.id)])
#         if os.path.exists(weight_tdi_path):
#             neuron.link_weights_tdi(weights_tdi[str(neuron.id)])
#
#     return weights
