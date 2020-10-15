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


def generate_pdf(spinet, rows, cols, nb_synapses, nb_layers, layer):
    images = natsorted(os.listdir(spinet.path+"images/simple_cells/"))
    pdf = FPDF("P", "mm", (cols*11, rows*11*nb_synapses))
    pdf.add_page()
    
    count = nb_synapses * layer
    for i in range(cols):
        for j in range(rows):
            for s in range(nb_synapses):
                pdf.image(spinet.path+"images/simple_cells/"+images[count], x=i*11, y=j*11*nb_synapses+s*10.4, w=10, h=10)
                count += 1
            count += nb_synapses * (nb_layers - 1)
    return pdf


def generate_pdf_layers(spinet, rows, cols, nb_synapses, nb_layers):
    images = natsorted(os.listdir(spinet.path+"images/simple_cells/"))
    pdf = FPDF("P", "mm", (cols*11, rows*11*nb_layers))
    pdf.add_page()
    
    count = 0
    for i in range(cols):
        for j in range(rows):
            for l in range(nb_layers):
                pdf.image(spinet.path+"images/simple_cells/"+images[count], x=i*11, y=j*11*nb_layers+l*10.4, w=10, h=10)
                count += nb_synapses
    return pdf


def generate_pdf_weight_sharing(spinet):
    images = natsorted(os.listdir(spinet.path+"images/simple_cells/"))

    selection = []
    for i in range(0, len(images), len(spinet.l1xanchor)*len(spinet.l1yanchor)*spinet.l1depth):
        selection += images[i:(i+1)+(spinet.l1depth-1)]

    nb_blocs_side = len(spinet.l1xanchor)
    size_im = 11
    nb_im_per_bloc = int(np.sqrt(spinet.l1depth))
    bloc_size = nb_im_per_bloc * 11

    pdf = FPDF("P", "mm", (nb_blocs_side*bloc_size+30, nb_blocs_side*bloc_size+30))
    pdf.add_page()

    count = 0
    for row in range(nb_blocs_side):
        for col in range(nb_blocs_side):
            for xim in range(nb_im_per_bloc):
                for yim in range(nb_im_per_bloc):
                    pdf.image(spinet.path+"images/simple_cells/"+selection[count], x=row*(bloc_size+10) + xim*size_im,
                              y=col*(bloc_size+10) + yim*size_im, w=10, h=10)
                    count += 1
    return pdf


def generate_pdf_complex_cell(spinet, layer):    
    pdf = FPDF("P", "mm", (len(spinet.l1xanchor)*spinet.l1width*11 + len(spinet.l1xanchor)*11, len(spinet.l1yanchor)*spinet.l1height*spinet.l1depth*11 + len(spinet.l1yanchor)*11))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "")

    for c, complex_cell in enumerate(spinet.complex_cells):
        xc, yc, zc = complex_cell.params["position"]
        ox, oy, oz = complex_cell.params["offset"]
            
        if zc == layer:
            maximum = np.max(complex_cell.weights)            
            for z, k in enumerate(sort_connections(spinet, complex_cell, oz)):
                for i in range(ox, ox + spinet.l1width):
                    for j in range(oy, oy + spinet.l1height):
                        simple_cell = spinet.simple_cells[spinet.layout1[i, j, k]]
                        xs, ys, zs = simple_cell.params["position"]
                        
                        weight_sc = complex_cell.weights[k, ys - oy, xs - ox] / maximum
                        img = weight_sc * np.array(Image.open(simple_cell.weight_images[0]))
                        path = spinet.path+"images/complex_connections/"+str(c)+"_simple_"+str(spinet.layout1[i, j, k])+".png"
                        Image.fromarray(img.astype('uint8')).save(path)
                        
                        pos_x = xc * (11 * spinet.l1width + 10) + (xs - ox) * 11
                        pos_y = yc * (11 * spinet.l1height * spinet.l1depth + spinet.l1depth * 2 + 10) + z * (11 * spinet.l1height + 2) + (ys - oy) * 11
                        pdf.image(path, x=pos_x, y=pos_y, w=10, h=10)
    return pdf

def sort_connections(spinet, complex_cell, oz):
    strengths = []
    for i in range(oz, oz + spinet.l1depth):
        strengths.append(np.sum(complex_cell.weights[i]))
    return np.argsort(strengths)[::-1]

def load_array_param(spinet, param):
    simple_array = np.zeros(spinet.nb_simple_cells)
    for i, simple_cell in enumerate(spinet.simple_cells):
        simple_array[i] = simple_cell.params[param]
    return simple_array, 0
    simple_array = simple_array.reshape((len(spinet.l1xanchor) * spinet.l1width, len(spinet.l1yanchor) * spinet.l1height, spinet.l1depth)).transpose((1, 0, 2))
    
    complex_array = np.zeros(spinet.nb_complex_cells)
    for i, complex_cell in enumerate(spinet.complex_cells):
        complex_array[i] = complex_cell.params[param]
    complex_array = complex_array.reshape((len(spinet.l2xanchor) * spinet.l2width, len(spinet.l2yanchor) * spinet.l2height, spinet.l2depth)).transpose((1, 0, 2))
    
    return simple_array, complex_array


def display_network(spinets, pooling=0):
    for spinet in spinets:
        spinet.generate_weight_images()
                
        if spinet.weight_sharing:
            pdf = generate_pdf_weight_sharing(spinet)
            pdf.output(spinet.path+"figures/weight_sharing.pdf", "F")
        else:
            for layer in range(spinet.l1depth):
                pdf = generate_pdf(spinet, spinet.l1height, spinet.l1width, spinet.neuron1_synapses, spinet.l1depth, layer)
                pdf.output(spinet.path+"figures/"+str(layer)+".pdf", "F")
            pdf = generate_pdf_layers(spinet, spinet.l1height, spinet.l1width, spinet.neuron1_synapses, spinet.l1depth)
            pdf.output(spinet.path+"figures/multi_layer.pdf", "F")

        if pooling:
            for layer in range(spinet.l2depth):
                pdf = generate_pdf_complex_cell(spinet, layer)
                pdf.output(spinet.path+"figures/complex_figures/"+str(layer)+".pdf", "F")
        
