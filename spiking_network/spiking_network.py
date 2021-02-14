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
import itertools

from aedat_tools.aedat_tools import load_params, delete_files

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
        self.unpack_json(path+"configs/network_config.json")
        self.neurons = []
        self.simple_cells = []
        self.complex_cells = []
        
        self.sspikes = []
        self.cspikes = []
        
        neurons_paths = natsorted(os.listdir(path + "weights/complex_cells"))
        for paths in [neurons_paths[i:i+2] for i in range(0, len(neurons_paths), 2)]:
            neuron = Neuron("complex", path+"configs/complex_cell_config.json", path+"weights/complex_cells/", *paths)
            self.neurons.append(neuron)
            self.complex_cells.append(neuron)
            if neuron.tracking:
                self.cspikes.append(neuron.spike_train)
        
        neurons_paths = natsorted(os.listdir(path + "weights/simple_cells"))
        for paths in [neurons_paths[i:i+2] for i in range(0, len(neurons_paths), 2)]:
            neuron = Neuron("simple", path+"configs/simple_cell_config.json", path+"weights/simple_cells/", *paths)
            self.neurons.append(neuron)
            self.simple_cells.append(neuron)
            
            if neuron.tracking:
                self.sspikes.append(neuron.spike_train)
                
        self.nb_neurons = len(self.neurons)
        self.nb_simple_cells = len(self.simple_cells)
        self.nb_complex_cells = len(self.complex_cells)
        
        self.layout1 = np.load(path + "weights/layout1.npy")
        # self.layout2 = np.load(path + "weights/layout2.npy")
        
        if np.array(self.sspikes).size > 0:
            self.sspikes = np.array(list(itertools.zip_longest(*self.sspikes, fillvalue=0))).T
            self.sspikes[self.sspikes != 0] -= np.min(self.sspikes[self.sspikes != 0])
        if np.array(self.cspikes).size > 0:
            self.cspikes = np.array(list(itertools.zip_longest(*self.cspikes, fillvalue=0))).T
            self.cspikes[self.cspikes != 0] -= np.min(self.cspikes[self.cspikes != 0])
            
        if os.path.exists(self.path+"gabors/data/direction_response.npy"):
            self.directions = np.load(self.path+"gabors/data/direction_response.npy")
            self.orientations = self.directions[0:8] + self.directions[8:16]
        
    def generate_weight_images(self):
        for i, neuron in enumerate(self.simple_cells):
            for synapse in range(self.neuron1_synapses):
                for camera in range(self.nb_cameras):
                    weights = reshape_weights(neuron.weights[:, camera, synapse], self.neuron1_width, self.neuron1_height)
                    path = self.path+"images/simple_cells/"+str(i)+"_syn"+str(synapse)+"_cam"+str(camera)+".png"
                    compress_weight(weights, path)
                    neuron.weight_images.append(path)

        for i, neuron in enumerate(self.complex_cells):
            for lay in range(self.neuron2_depth):
                dim = np.zeros((self.neuron2_width, self.neuron2_height))
                weight = np.stack((neuron.weights[:, :, lay], dim, dim), axis=2)
                path = self.path+"images/complex_cells/"+str(i)+"_lay_"+str(lay)+".png"
                compress_weight(np.kron(weight, np.ones((7, 7, 1))), path)
                neuron.weight_images.append(path)


    def generate_weight_mat(self):
        weights = []
        if self.weight_sharing:
            weights = [neuron.weights for neuron in self.simple_cells[0:self.l1depth]]
            # for i in range(0, self.nb_simple_cells, self.l1depth*self.l1width*self.l1height):
            #     weights += [neuron.weights for neuron in self.simple_cells[i:i+self.l1depth]]
        else:
            weights = [neuron.weights for neuron in self.simple_cells]

        basis = np.zeros((200, len(weights)))

        for i, weight in enumerate(weights):    
            basi = (weight[0, 0, 0] - weight[1, 0, 0]).flatten("F")
            basis[0:100, i] = basi
            basis[100:200, i] = basi
        
        sio.savemat(self.path+"gabors/data/weights.mat", {"data": basis})
        
    def unpack_json(self, json_path):
        json = load_params(json_path)
        self.weight_sharing = json["WeightSharing"]
        self.l1width = json["L1Width"]
        self.l1height = json["L1Height"]
        self.l1depth = json["L1Depth"]
        self.l1xanchor = json["L1XAnchor"]
        self.l1yanchor = json["L1YAnchor"]
        self.neuron1_width = json["Neuron1Width"]
        self.neuron1_height = json["Neuron1Height"]
        self.neuron1_synapses = json["Neuron1Synapses"]
        
        self.l2width = json["L2Width"]
        self.l2height = json["L2Height"]
        self.l2depth = json["L2Depth"]
        self.l2xanchor = json["L2XAnchor"]
        self.l2yanchor = json["L2YAnchor"]
        self.neuron2_width = json["Neuron2Width"]
        self.neuron2_height = json["Neuron2Height"]
        self.neuron2_depth = json["Neuron2Depth"]
        self.nb_cameras = json["NbCameras"]
        
    def clean_network(self, simple_cells, complex_cells, json_only):
        if json_only:
            if simple_cells:
                for file in os.listdir(self.path+"weights/simple_cells/"):
                    if file.endswith(".json"):
                        os.remove(self.path+"weights/simple_cells/"+file)
            if complex_cells:
                for file in os.listdir(self.path+"weights/complex_cells/"):
                    if file.endswith(".json"):
                        os.remove(self.path+"weights/complex_cells/"+file)
        else:
            if simple_cells:
                delete_files(self.path+"weights/simple_cells/")
            if complex_cells:
                delete_files(self.path+"weights/complex_cells/")
            delete_files(self.path+"images/complex_connections/")
            delete_files(self.path+"images/simple_cells/")
            delete_files(self.path+"images/complex_cells/")
        
    def save_complex_directions(self, spikes, rotations):
        spike_vector = []
        for rot in range(rotations.size):
            spike_vector.append(np.count_nonzero(spikes[rot], axis=1))
        spike_vector = np.array(spike_vector)
        
        np.save(self.path+"gabors/data/direction_response", spike_vector)
        self.directions = spike_vector
        self.orientations = self.directions[0:8] + self.directions[8:16]

class Neuron:
    """Spiking Neuron class"""
    
    def __init__(self, neuron_type, config_path, path, param_path, weight_path):
        self.type = neuron_type
        self.unpack_json_config(config_path)
        self.weights = np.load(path + weight_path)
        self.unpack_json_params(path + param_path)
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
        
    def unpack_json_config(self, json_path):
        json = load_params(json_path)
        if self.type == "simple":
            self.min_thresh = json["MIN_THRESH"]
            self.delta_sra = json["ETA_SRA"]
            self.delta_sr = json["ETA_TA"]
            self.tau_sra = json["TAU_SRA"]
            self.target_spike_rate = json["TARGET_SPIKE_RATE"]
            self.synapse_delay = json["SYNAPSE_DELAY"]
            
        self.vthresh = json["VTHRESH"]
        self.vreset = json["VRESET"]
        self.delta_rp = json["ETA_RP"]
        self.tau_rp = json["TAU_RP"]
        self.tau_m = json["TAU_M"]
        self.tau_ltp = json["TAU_LTP"]
        self.tau_ltd= json["TAU_LTD"]
        self.norm_factor = json["NORM_FACTOR"]
        self.delta_vp = json["ETA_LTP"]
        self.delta_vd = json["ETA_LTD"]
        self.delta_inh = json["ETA_INH"]
        self.decay_factor = json["DECAY_FACTOR"]
        self.stdp_learning = json["STDP_LEARNING"]
        self.tracking = json["TRACKING"]
        
    def unpack_json_params(self, json_path):
        json = load_params(json_path)
        self.count_spike = json["count_spike"]
        self.creation_time = json["creation_time"]
        self.in_connections = json["in_connections"]
        self.learning_decay = json["learning_decay"]
        self.offset = json["offset"]
        self.out_connections = json["out_connections"]
        self.position = json["position"]
        self.potential = json["potential"]
        self.potential_train = json["potential_train"]
        self.recent_spikes = json["recent_spikes"]
        self.spike_train = json["spike_train"]
        self.spiking_rate = json["spiking_rate"]
        self.threshold = json["threshold"]
        self.inhibition_connections = json["inhibition_connections"]