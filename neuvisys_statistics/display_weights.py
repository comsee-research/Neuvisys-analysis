#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:47:43 2020

@author: thomas
"""

import json
import os
import numpy as np
from PIL import Image
from natsort import natsorted
from fpdf import FPDF


def compress_weight(weights, path):
    img = np.array(255 * (weights / weights.max()), dtype=np.uint8)
    img = Image.fromarray(img).save(path)


def load_params(param_path):
    with open(param_path) as file:
        return json.load(file)


def load_neurons_infos(neuron_path):
    files = natsorted([neuron_path+f for f in os.listdir(neuron_path) if f.endswith(".json")])
    infos = []
    for file in files:
        with open(file) as f:
            jayson = json.load(f)
        infos.append({"threshold": jayson["threshold"], "spiking_rate": jayson["spiking_rate"]})
    return infos


def plot_network(directory):
    files = natsorted([directory+"weights/"+f for f in os.listdir(directory+"weights/") if f.endswith(".npy")])
    network_params = {**load_params(directory+"configs/neuron_config.json"), **load_params(directory+"configs/network_config.json")}

    neurons_info = load_neurons_infos(directory+"weights/")
    neurons_info = [str(int(info["threshold"])) + "|" + str(round(info["spiking_rate"], 1)) for info in neurons_info]
    
    for i, file in enumerate(files):
        if "pooling" in file:
            weights = np.load(file)
            for lay in range(network_params["L1Depth"]):
                dim = np.zeros((network_params["Neuron2Width"], network_params["Neuron2Height"]))
                weight = np.stack((weights[lay], dim, dim), axis=2)
                compress_weight(np.kron(weight, np.ones((7, 7, 1))), directory+"images/pooling_"+str(i)+"_lay_"+str(lay)+".png")
        else:
            for synapse in range(network_params["Neuron1Synapses"]):
                weights = np.moveaxis(np.concatenate((np.load(file)[:, synapse], np.zeros((1, network_params["Neuron1Width"], network_params["Neuron1Height"]))), axis=0), 0, 2)
                compress_weight(np.kron(weights, np.ones((3, 3, 1))), directory+"images/"+str(i)+"_syn"+str(synapse)+".png")

    return network_params


def generate_pdf(directory, title, rows, cols, nb_synapses, nb_layers, layer):
    header = 30
    images = natsorted([f for f in os.listdir(directory) if "pooling" not in f])
    pdf = FPDF("P", "mm", (cols*11, header+rows*11*nb_synapses))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, title)
    
    count = nb_synapses * layer
    for i in range(cols):
        for j in range(rows):
            for s in range(nb_synapses):
                pdf.image(directory+images[count], x=i*11, y=header+j*11*nb_synapses+s*10.4, w=10, h=10)
                count += 1
            count += nb_synapses * (nb_layers - 1)
    return pdf

def generate_pdf_pooling(directory, title, rows, cols, depth):
    header = 30
    images = natsorted([f for f in os.listdir(directory) if "pooling" in f])
    pdf = FPDF("P", "mm", (cols*11, header+rows*11*depth))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    
    count = 0
    for i in range(cols):
        for j in range(rows):
            for k in range(depth):
                pdf.image(directory+images[count], x=i*11, y=header+j*11*depth+k*10.4, w=10, h=10)
                count += 1
    return pdf


def generate_pdf_layers(directory, title, rows, cols, nb_synapses, nb_layers):
    header = 30
    images = natsorted(os.listdir(directory))
    pdf = FPDF("P", "mm", (cols*11, header+rows*11*nb_layers))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, title)
    
    count = 0
    for i in range(cols):
        for j in range(rows):
            for l in range(nb_layers):
                pdf.image(directory+images[count], x=i*11, y=header+j*11*nb_layers+l*10.4, w=10, h=10)
                count += nb_synapses
    return pdf


def generate_pdf_weight_sharing(directory, title, rows, cols, nb_synapses, nb_layers):
    header = 30
    images = natsorted(os.listdir(directory))
    pdf = FPDF("P", "mm", (cols*11, header+3*11*nb_layers))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, title)
    
    count = 0
    for i in range(3):
        for j in range(3):
            for l in range(nb_layers):
                pdf.image(directory+images[count], x=100+i*31, y=header+j*11*nb_layers+l*10.4, w=10, h=10)
                count += nb_synapses
            count += 16*nb_layers
    return pdf


directory = "/home/thomas/neuvisys-dv/configuration/network/"
network_params = plot_network(directory)
for layer in range(network_params["L1Depth"]):
    pdf = generate_pdf(directory+"images/", str(network_params), network_params["L1Height"], network_params["L1Width"], network_params["Neuron1Synapses"], network_params["L1Depth"], layer)
    pdf.output(directory+"figures/"+str(layer)+".pdf", "F")
pdf = generate_pdf_pooling(directory+"images/", str(network_params), network_params["L2Height"], network_params["L2Width"], network_params["L1Depth"])
pdf.output(directory+"figures/pooling.pdf", "F")
# for i in range(10):
#     directory = "/home/thomas/neuvisys-dv/configuration/network_"+str(i)+"/"
    
#     network_params = plot_network(directory)
#     # for layer in range(network_params["L1Depth"]):
#         # pdf = generate_pdf(directory+"images/", str(network_params), network_params["L1Height"], network_params["L1Width"], network_params["Neuron1Synapses"], network_params["L1Depth"], layer)
#         # pdf.output(directory+"figures/"+str(layer)+".pdf", "F")
        
#     pdf = generate_pdf_weight_sharing(directory+"images/", str(network_params), network_params["L1Height"], network_params["L1Width"], network_params["Neuron1Synapses"], network_params["L1Depth"])
#     pdf.output(directory+"figures/multi_layer.pdf", "F")