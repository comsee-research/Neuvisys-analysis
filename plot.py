#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:27:10 2020

@author: thomas
"""

import os
from natsort import natsorted, ns

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd

def figure(rows, cols, height, width, title, subplot_title, xaxis_title, yaxis_title, plotfunction, **kwargs):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_title, x_title=xaxis_title, y_title=yaxis_title, vertical_spacing=0.07, horizontal_spacing=0.07)
    for i in range(rows):
        for j in range(cols):
            fig.add_trace(plotfunction({key: value[i*cols+j] for key, value in kwargs.items()}), row=i+1, col=j+1)
    
    # fig.update_traces(dict(showscale=False))
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"),
        height=height,
        width=width,
        title_text=title)
    return fig

pio.renderers.default = "firefox"

#%%
directory = "data/noise_without_frames/"
files = natsorted(os.listdir(directory))
correlations = [np.load(directory + file) for file in files]
title = "Spatial Correlation (%) of a noise recording without frames activated"
xaxis_title= "pixels"
yaxis_title= "pixels"
names = [text.split("_")[2] + "μs range" for text in files]

figure(3, 3, 1200, 1200, title, names, xaxis_title, yaxis_title, go.Heatmap, z=correlations).show()

directory = "data/noise_with_frames/"
files = natsorted(os.listdir(directory))
correlations = [np.load(directory + file) for file in files]
title = "Spatial Correlation (%) of a noise recording with frames activated"

figure(3, 3, 1200, 1200, title, names, xaxis_title, yaxis_title, go.Heatmap, z=correlations).show()

#%%
directory = "/home/thomas/neuvisys-dv/results/weights_2/"
files = natsorted(os.listdir(directory))
numberSpikes = np.load(directory + files[-1])
files = files[:-1]
correlations = [np.moveaxis(np.concatenate((np.load(directory + file), np.zeros((1, 10, 10))), axis=0), 0, 2) for file in files]

rows = 18
cols = 12
layers = 4
title = "Weights"
xaxis_title= "pixels"
yaxis_title= "pixels"

for k in range(layers):
    data = correlations[k::layers]
    names = numberSpikes[k::layers].astype(str)
    data = [dat/dat.max() for dat in data]
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, x_title=xaxis_title, y_title=yaxis_title, vertical_spacing=0.015, horizontal_spacing=0.005)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"),
        height=2700,
        width=1800,
        title_text=title)
    for i in range(rows):
        for j in range(cols):
            fig.add_trace(px.imshow(data[i*cols+j])['data'][0], i+1, j+1)
    
    fig.show()