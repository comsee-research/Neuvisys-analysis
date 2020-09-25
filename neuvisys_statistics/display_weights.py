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
from spiking_network import SpikingNetwork


def load_neurons_infos(neuron_path):
    files = natsorted([neuron_path+f for f in os.listdir(neuron_path) if f.endswith(".json")])
    infos = []
    for file in files:
        with open(file) as f:
            jayson = json.load(f)
        infos.append({"threshold": jayson["threshold"], "spiking_rate": jayson["spiking_rate"]})
    return infos


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


def generate_pdf_weight_sharing(directory, title, nb_synapses, bloc_width, bloc_height, depth):
    header = 0
    images = natsorted(os.listdir(directory))

    selection = []
    for i in range(0, len(images), bloc_width*bloc_height*depth):
        selection += images[i:(i+1)+(depth-1)]

    nb_blocs_side = 3
    size_im = 11
    nb_im_per_bloc = int(np.sqrt(depth))
    bloc_size = nb_im_per_bloc * 11

    pdf = FPDF("P", "mm", (nb_blocs_side*bloc_size+30, header+nb_blocs_side*bloc_size+30))
    pdf.add_page()

    pdf.set_font('Arial', '', 10)
    # pdf.multi_cell(0, 5, title)

    count = 0
    for row in range(nb_blocs_side):
        for col in range(nb_blocs_side):
            for xim in range(nb_im_per_bloc):
                for yim in range(nb_im_per_bloc):
                    pdf.image(directory+selection[count], x=row*(bloc_size+10) + xim*size_im,
                              y=header + col*(bloc_size+10) + yim*size_im, w=10, h=10)
                    count += 1
    return pdf


def generate_pdf_complex_cell(spinet, layer1, layer2):
    cols = spinet.net_var["L1Width"] * 12
    rows = spinet.net_var["L1Height"] * 12
    
    pdf = FPDF("P", "mm", (cols, rows))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "")

    for i, complex_cell in enumerate(spinet.complex_cells):
        if complex_cell.params["position"][2] == layer2:
            maximum = np.max(complex_cell.weights[layer1])
            
            xc = complex_cell.params["position"][0]
            yc = complex_cell.params["position"][1]
            
            for neuron_ind in complex_cell.params["in_connections"]:
                simple_cell = spinet.simple_cells[neuron_ind]
                if simple_cell.params["position"][2] == layer1:
                    xs = simple_cell.params["position"][0]
                    ys = simple_cell.params["position"][1]
                    
                    dim = complex_cell.weights[layer1, ys - complex_cell.params["offset"][1], xs - complex_cell.params["offset"][0]]
                    img = (dim / maximum) * np.array(Image.open(simple_cell.weight_images[0]))
                    Image.fromarray(img.astype('uint8')).save(spinet.path + "temp/complex_"+str(i)+"_simple_"+str(neuron_ind)+".png")
                    
                    pos_x = 3 * xc + xs * 11
                    pos_y = 3 * yc + ys * 11
                    
                    pdf.image(spinet.path + "temp/complex_"+str(i)+"_simple_"+str(neuron_ind)+".png", x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def load_array_param(spinet, param):
    simple_array = np.zeros(spinet.nb_simple_cells)
    for i, simple_cell in enumerate(spinet.simple_cells):
        simple_array[i] = simple_cell.params[param]
    simple_array = simple_array.reshape((spinet.net_var["L1Width"], spinet.net_var["L1Height"], spinet.net_var["L1Depth"])).transpose((1, 0, 2))
    
    complex_array = np.zeros(spinet.nb_complex_cells)
    for i, complex_cell in enumerate(spinet.complex_cells):
        complex_array[i] = complex_cell.params[param]
    complex_array = complex_array.reshape((spinet.net_var["L2Width"], spinet.net_var["L2Height"], spinet.net_var["L2Depth"])).transpose((1, 0, 2))
    
    return simple_array, complex_array


def display_network(spinets, pooling=0):
    for spinet in spinets:
        spinet.generate_weight_images(spinet.path + "images/")
                
        if spinet.net_var["WeightSharing"]:
            pdf = generate_pdf_weight_sharing(spinet.path+"images/", str(spinet.net_var), spinet.net_var["Neuron1Synapses"], 4, 4, spinet.net_var["L1Depth"])
            pdf.output(spinet.path+"figures/weight_sharing.pdf", "F")
        else:
            for layer in range(spinet.net_var["L1Depth"]):
                pdf = generate_pdf(spinet.path+"images/", str(spinet.net_var), spinet.net_var["L1Height"], spinet.net_var["L1Width"], spinet.net_var["Neuron1Synapses"], spinet.net_var["L1Depth"], layer)
                pdf.output(spinet.path+"figures/"+str(layer)+".pdf", "F")
            pdf = generate_pdf_layers(spinet.path+"images/", str(spinet.net_var), spinet.net_var["L1Height"], spinet.net_var["L1Width"], spinet.net_var["Neuron1Synapses"], spinet.net_var["L1Depth"])
            pdf.output(spinet.path+"figures/multi_layer.pdf", "F")

        if pooling:
            for layer2 in range(spinet.net_var["L2Depth"]):
                # for layer1 in range(spinet.net_var["L1Depth"]):
                pdf = generate_pdf_complex_cell(spinet, 0, layer2)
                pdf.output(spinet.path+"figures/complex_cells_"+str(layer2)+".pdf", "F")
        
