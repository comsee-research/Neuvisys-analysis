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
directory = "/home/thomas/Bureau/test/"
files = natsorted(os.listdir(directory))
correlations = [np.moveaxis(np.concatenate((np.load(directory + file), np.zeros((1, 10, 10))), axis=0), 0, 2) for file in files]
title = "Weights"
xaxis_title= "pixels"
yaxis_title= "pixels"

fig = make_subplots(rows=5, cols=5, subplot_titles=files, x_title=xaxis_title, y_title=yaxis_title, vertical_spacing=0.07, horizontal_spacing=0.07)
for i in range(5):
    for j in range(5):
        fig.add_trace(go.Image(z=correlations[i*5+j], zmax=[1, 1, 1, 1]), i+1, j+1)

fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"),
    height=1500,
    width=1500,
    title_text=title)
fig.show()