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
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt

def generate_pdf_simple_cell(spinet, layer, camera):
    pdf = FPDF("P", "mm", (11*spinet.l1width, 11*spinet.l1height*spinet.neuron1_synapses))
    pdf.add_page()
    
    pos_x = 0
    pos_y = 0
    for neuron in spinet.simple_cells:
        x, y, z = neuron.position
        if z == layer:
            for i in range(spinet.neuron1_synapses):
                pos_x = x * 11
                pos_y = y * spinet.neuron1_synapses * 11 + i * 11
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
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


def generate_pdf_weight_sharing(spinet, camera):
    pdf = FPDF("P", "mm", (11*len(spinet.l1xanchor)*int(np.sqrt(spinet.l1depth)) + (len(spinet.l1xanchor)-1)*10, 11*len(spinet.l1yanchor)*int(np.sqrt(spinet.l1depth)) + (len(spinet.l1yanchor)-1)*10))
    pdf.add_page()
    
    pos_x = 0
    pos_y = 0
    shift = np.arange(spinet.l1depth).reshape((int(np.sqrt(spinet.l1depth)), int(np.sqrt(spinet.l1depth))))
    for i in range(0, spinet.nb_simple_cells, spinet.l1depth*spinet.l1width*spinet.l1height):
        for neuron in spinet.simple_cells[i:i+spinet.l1depth]:
            x, y, z = neuron.position
            pos_x = (x / spinet.l1width) * int(np.sqrt(spinet.l1depth)) * 11 + np.where(shift == z)[0][0] * 11 + (x / spinet.l1width) * 10
            pos_y = (y / spinet.l1height) * int(np.sqrt(spinet.l1depth)) * 11 + np.where(shift == z)[1][0] * 11 + (y / spinet.l1height) * 10
            pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def generate_pdf_complex_cell(spinet, layer):    
    pdf = FPDF("P", "mm", (len(spinet.l1xanchor)*spinet.l1width*11 + len(spinet.l1xanchor)*11, len(spinet.l1yanchor)*spinet.l1height*spinet.neuron2_depth*11 + len(spinet.l1yanchor)*11 + len(spinet.l1yanchor)*10))
    pdf.add_page()
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "")

    for c, complex_cell in enumerate(spinet.complex_cells):
        xc, yc, zc = complex_cell.position
        ox, oy, oz = complex_cell.offset
            
        if zc == layer:
            maximum = np.max(complex_cell.weights)            
            for z, k in enumerate(sort_connections(spinet, complex_cell, oz)):
                for i in range(ox, ox + spinet.l1width):
                    for j in range(oy, oy + spinet.l1height):
                        simple_cell = spinet.simple_cells[spinet.layout1[i, j, k]]
                        xs, ys, zs = simple_cell.position
                        
                        weight_sc = complex_cell.weights[xs - ox, ys - oy, k] / maximum
                        img = weight_sc * np.array(Image.open(simple_cell.weight_images[0]))
                        path = "/media/alphat/SSD Games/Thesis/temp/" + str(c)+"_simple_"+str(spinet.layout1[i, j, k])+".png"
                        Image.fromarray(img.astype('uint8')).save(path)
                        
                        pos_x = xc * (11 * spinet.l1width + 10) + (xs - ox) * 11
                        pos_y = yc * (11 * spinet.l1height * spinet.neuron2_depth + spinet.neuron2_depth * 2 + 10) + z * (11 * spinet.l1height + 2) + (ys - oy) * 11
                        pdf.image(path, x=pos_x, y=pos_y, w=10, h=10)
    return pdf

def sort_connections(spinet, complex_cell, oz):
    strengths = []
    for z in range(spinet.neuron2_depth):
        strengths.append(np.sum(complex_cell.weights[:, :, z]))
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

def complex_cells_directions(spinet, rotations):    
    plot_directions(spinet, spinet.directions, rotations)
    plot_orientations(spinet, spinet.orientations, rotations[0:8])

def plot_directions(spinet, directions, rotations):
    temp = np.zeros((directions.shape[0]+1, directions.shape[1]))
    temp[:-1, :] = directions
    temp[-1, :] = directions[0, :]
    angles = np.append(rotations, 0) * np.pi / 180
    spike_vector = temp
    
    for i in range(spike_vector.shape[1]):
        mean = mean_response(spike_vector[:-1, i], angles[:-1])
        
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.set_title("Cell"+str(i))
        ax.plot(angles, spike_vector[:, i], "r")
        ax.arrow(np.angle(mean), 0, 0, 2*np.abs(mean), width=0.015, head_width=1, head_length=1, length_includes_head=True, edgecolor='black', facecolor='black', lw=2, zorder=5)
        ax.set_thetamax(360)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.savefig(spinet.path+"figures/complex_directions/"+str(i))

def plot_orientations(spinet, orientations, rotations):
    temp = np.zeros((orientations.shape[0]+1, orientations.shape[1]))
    temp[:-1, :] = orientations
    temp[-1, :] = orientations[0, :]
    angles = np.append(rotations, 180) * np.pi / 180
    spike_vector = temp
    
    for i in range(spike_vector.shape[1]):
        mean = mean_response(spike_vector[:-1, i], angles[:-1])
        
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.set_title("Cell"+str(i))
        ax.plot(angles, spike_vector[:, i], "r")
        ax.arrow(np.angle(mean), 0, 0, np.abs(mean), width=0.015, head_width=1, head_length=1, length_includes_head=True, edgecolor='black', facecolor='black', lw=2, zorder=5)
        ax.set_thetamax(180)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.savefig(spinet.path+"figures/complex_orientations/"+str(i))
        
def mean_response(directions, angles):
    return np.mean(directions * np.exp(1j * angles))

def display_network(spinets, pooling=0):
    for spinet in spinets:
        spinet.generate_weight_images()
                
        if spinet.weight_sharing:
            for i in range(spinet.nb_cameras):
                pdf = generate_pdf_weight_sharing(spinet, i)
                pdf.output(spinet.path+"figures/simple_figures/weight_sharing_"+str(i)+".pdf", "F")
        else:
            for i in range(spinet.nb_cameras):
                for layer in range(spinet.l1depth):
                    pdf = generate_pdf_simple_cell(spinet, layer, i)
                    pdf.output(spinet.path+"figures/simple_figures/"+str(layer)+"_"+str(i)+".pdf", "F")
            # pdf = generate_pdf_layers(spinet, spinet.l1height, spinet.l1width, spinet.neuron1_synapses, spinet.l1depth)
            # pdf.output(spinet.path+"figures/multi_layer.pdf", "F")

        if pooling:
            p = Pool(10)
            args = [(spinet, x) for x in list(np.arange(spinet.l2depth))]
            pdfs = p.starmap(generate_pdf_complex_cell, args)

            proc = []
            for i, pdf in enumerate(pdfs[0:4]):
                process = Process(target=pdf.output, args=(spinet.path+"figures/complex_figures/"+str(i)+".pdf", "F"))
                process.start()
                proc.append(process)
            for process in proc:
                process.join()

