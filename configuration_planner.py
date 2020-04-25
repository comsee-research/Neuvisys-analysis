#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:17 2020

@author: thomas
"""

# "DELTA_VP": norm(loc=0.065, scale=0.015),
# "DELTA_VD": norm(loc=0.025, scale=0.015),
# "TAU_LTP": norm(loc=10000, scale=5000),
# "TAU_LTD": norm(loc=15000, scale=5000),
# "VTHRESH": norm(loc=17, scale=4.5),
# "VRESET": [-20],
# "TAU_M": norm(loc=12000, scale=6000),
# "TAU_INHIB": norm(loc=8500, scale=3500),

from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import subprocess
import os
import numpy as np
from PIL import Image
from natsort import natsorted
from scipy.stats import norm

from fpdf import FPDF

def generate_pdf(directory, title, rows, cols, nb_synapses, nb_layers, layer):
    header = 30
    images = natsorted(os.listdir(directory))
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

def plot_network(directory, network_id):
    files = natsorted([directory+"weights/"+str(network_id)+"/"+f for f in os.listdir(directory+"weights/"+str(network_id)+"/") if f.endswith(".npy")])
    network_params = load_params(directory+"configs/"+str(network_id)+".json")
    
    neurons_info = load_neurons_infos(directory+"weights/" + str(network_id) + "/")
    neurons_info = [str(int(info["threshold"])) + "|" + str(round(info["spiking_rate"], 1)) for info in neurons_info]
    
    try:
        os.mkdir(directory+"images/"+str(network_id)+"/")
    except:
        pass
    
    for i, file in enumerate(files):
        for synapse in range(network_params["NEURON_SYNAPSES"]):
            if network_params["NEURON_SYNAPSES"] < 2:
                weights = np.moveaxis(np.concatenate((np.load(file), np.zeros((1, network_params["NEURON_WIDTH"], network_params["NEURON_HEIGHT"]))), axis=0), 0, 2)
            else:
                weights = np.moveaxis(np.concatenate((np.load(file)[:, synapse], np.zeros((1, network_params["NEURON_WIDTH"], network_params["NEURON_HEIGHT"]))), axis=0), 0, 2)     
                
            weights = np.kron(weights, np.ones((3, 3, 1)))
            compress_weight(weights, directory+"images/"+str(network_id)+"/"+str(i)+"_syn"+str(synapse)+".png")
    return network_params

    # sizes = [os.path.getsize(directory+"images/"+str(network_id)+"/"+file) for file in natsorted(os.listdir(directory+"images/"+str(network_id))) if file.find("syn"+str(synapse)) != -1]

def generate_multiple_configurations(directory, sampler, n_iter):
    for i in range(n_iter):
        with open(directory+"configs/"+str(i)+".json", "w") as file:
            json.dump(sampler[i], file)

        os.mkdir(directory+"weights/"+str(i))
        with open("/home/thomas/neuvisys-dv/configs/conf.json", "w") as conf:
            json.dump({"SAVE_DATA": True,
                       "SAVE_DATA_LOCATION": directory+"weights/"+str(i)+"/",
                       "CONF_FILES_LOCATION": directory+"configs/"+str(i)+".json"}, conf)
    
        try:
            subprocess.run(["dv-runtime", "-b0"], timeout=190)
        except:
            print("Finished learning: " + str(i))


param_grid = {"NEURON_WIDTH": [10], "NEURON_HEIGHT": [10], "NEURON_SYNAPSES": [2, 3], "SYNAPSE_DELAY": [5000, 10000, 15000, 20000, 25000], "X_ANCHOR_POINT": [0], "Y_ANCHOR_POINT": [0], "NETWORK_WIDTH": [34], "NETWORK_HEIGHT": [26], "NETWORK_DEPTH": [1], "DELTA_VP": [0.06], "DELTA_VD": [0.02], "DELTA_SR": [0.1], "TAU_LTP": [10000], "TAU_LTD": [20000], "VTHRESH": [20, 40, 60, 80], "VRESET": [-10], "TAU_M": [10000], "TAU_INHIB": [5000], "NORM_FACTOR": [4], "NORM_THRESHOLD": [4], "TARGET_SPIKE_RATE": [0.5]}
    
directory = "/home/thomas/neuvisys-dv/configuration/"
sampler = list(ParameterGrid(param_grid))
n_iter = len(sampler)

for network_id in range(1):
    network_params = plot_network(directory, network_id)
    for layer in range(network_params["NETWORK_DEPTH"]):
        pdf = generate_pdf(directory+"/images/"+str(network_id)+"/", str(network_params), network_params["NETWORK_HEIGHT"], network_params["NETWORK_WIDTH"], network_params["NEURON_SYNAPSES"], network_params["NETWORK_DEPTH"], layer)
        pdf.output(directory+"/figures/"+str(network_id)+"_"+str(layer)+".pdf", "F")
        
    pdf = generate_pdf_layers(directory+"/images/"+str(network_id)+"/", str(network_params), network_params["NETWORK_HEIGHT"], network_params["NETWORK_WIDTH"], network_params["NEURON_SYNAPSES"], network_params["NETWORK_DEPTH"])
    pdf.output(directory+"/figures/"+str(network_id)+"_multi_layer.pdf", "F")
# generate_multiple_configurations(directory, sampler, n_iter)
