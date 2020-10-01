#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:32:36 2020

@author: thomas
"""

from natsort import natsorted
import os
from PIL import Image
import scipy.io as sio
import numpy as np

from aedat_tools.aedat_tools import load_params, delete_files

def compress_weight(weights, path):
    img = np.array(255 * (weights / weights.max()), dtype=np.uint8)
    img = Image.fromarray(img).save(path)

class SpikingNetwork:
    """Spiking Neural Network class"""
    
    def __init__(self, path):
        self.path = path
        self.net_var = load_params(path + "configs/network_config.json")
        self.neu_var = load_params(path + "configs/neuron_config.json")
        self.pool_neur_var = load_params(path + "configs/pooling_neuron_config.json")
        
        self.neurons = []
        self.simple_cells = []
        self.complex_cells = []
        neurons_paths = natsorted(os.listdir(path + "weights/complex_cells"))
        for paths in [neurons_paths[i:i+2] for i in range(0, len(neurons_paths), 2)]:
            neuron = Neuron("complex", path + "weights/complex_cells/", *paths)
            self.neurons.append(neuron)
            self.complex_cells.append(neuron)
        
        neurons_paths = natsorted(os.listdir(path + "weights/simple_cells"))
        for paths in [neurons_paths[i:i+2] for i in range(0, len(neurons_paths), 2)]:
            neuron = Neuron("simple", path + "weights/simple_cells/", *paths)
            self.neurons.append(neuron)
            self.simple_cells.append(neuron)
                
        self.nb_neurons = len(self.neurons)
        self.nb_simple_cells = len(self.simple_cells)
        self.nb_complex_cells = len(self.complex_cells)
        
        if self.net_var["WeightSharing"]:
            self.shared_weights = []
            for i in range(0, self.nb_neurons, 4*4*self.net_var["L1Depth"]):
                self.shared_weights += self.neurons[i:i+(self.net_var["L1Depth"])]
            self.shared_weights = [neuron.weights for neuron in self.shared_weights]
        
    def generate_weight_images(self):
        for i, neuron in enumerate(self.neurons):
            if neuron.type == "complex":
                for lay in range(self.net_var["L1Depth"]):
                    dim = np.zeros((self.net_var["Neuron2Width"], self.net_var["Neuron2Height"]))
                    weight = np.stack((neuron.weights[lay], dim, dim), axis=2)
                    path = self.path+"images/complex_cells/"+str(i)+"_lay_"+str(lay)+".png"
                    compress_weight(np.kron(weight, np.ones((7, 7, 1))), path)
                    neuron.weight_images.append(path)
            else:
                for synapse in range(self.net_var["Neuron1Synapses"]):
                    weights = np.moveaxis(np.concatenate((neuron.weights[:, synapse], np.zeros((1, self.net_var["Neuron1Width"], self.net_var["Neuron1Height"]))), axis=0), 0, 2)
                    path = self.path+"images/simple_cells/"+str(i)+"_syn"+str(synapse)+".png"
                    compress_weight(np.kron(weights, np.ones((3, 3, 1))), path)
                    neuron.weight_images.append(path)

    def generate_weight_mat(self, dest):
        if self.net_var["WeightSharing"]:
            basis = np.zeros((200, len(self.shared_weights)))
            weights = self.shared_weights
        else:
            basis = np.zeros((200, self.nb_neurons))
            weights = [neuron.weights for neuron in self.neurons]
        
        for i, weight in enumerate(weights):    
            basi = (weight[0, 0] - weight[1, 0]).flatten("F")
            basis[0:100, i] = basi
            basis[100:200, i] = basi
        sio.savemat(dest, {"data": basis})
        
    def clean_network(self):
        delete_files(self.path+"images/simple_cells/")
        delete_files(self.path+"images/complex_cells/")
        delete_files(self.path+"images/complex_connections/")
        delete_files(self.path+"weights/complex_cells/")
        delete_files(self.path+"weights/simple_cells/")

class Neuron:
    """Spiking Neuron class"""
    
    def __init__(self, neuron_type, path, param_path, weight_path):
        self.type = neuron_type
        self.weights = np.load(path + weight_path)
        self.params = load_params(path + param_path)
        self.weight_images = []
