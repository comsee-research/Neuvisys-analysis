#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:27:10 2020

@author: thomas
"""

import os
from natsort import natsorted
import json

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd

pio.renderers.default = "firefox"

    
#%%

img = np.load(directory + "neuron_155.npy")[1]
px.imshow(img).show()
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
px.imshow(magnitude_spectrum).show()

#%%

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
