#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:26:24 2020

@author: thomas
"""

import os
from natsort import natsorted
import json
from multiprocessing import Pool

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#%%
# synapse = 1

def create_plots(config):
    with open("/home/thomas/neuvisys-analysis/results/config_files/conf_"+str(config)+".json") as file:
        params = json.load(file)
    
    directory = "/home/thomas/neuvisys-analysis/results/weights/" + str(config) + "/"
    files = natsorted([f for f in os.listdir(directory) if f.endswith(".npy")])
    correlations = [np.moveaxis(np.concatenate((np.load(directory + file), np.zeros((1, params["NEURON_WIDTH"], params["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    # correlations = [np.moveaxis(np.concatenate((np.load(directory + file)[:, synapse], np.zeros((1, params["NEURON_WIDTH"], params["NEURON_HEIGHT"]))), axis=0), 0, 2) for file in files]
    
    xaxis_title= "pixels"
    yaxis_title= "pixels"
    
    for k in range(1):
        annot = str(params)
        title = annot[:len(annot)//2] + "<br>" + annot[len(annot)//2:]
        data = correlations[k::params["NETWORK_DEPTH"]]
        data = [dat/dat.max() for dat in data]
        
        fig = make_subplots(rows=params["NETWORK_HEIGHT"], cols=params["NETWORK_WIDTH"], x_title=xaxis_title, y_title=yaxis_title, vertical_spacing=0.015, horizontal_spacing=0.005)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        
        fig.update_layout(
            font=dict(
                family="Courier New, monospace",
                size=13,
                color="#7f7f7f"),
            height=2700,
            width=1800,
            title_text=title)
        
        for i in range(params["NETWORK_HEIGHT"]):
            for j in range(params["NETWORK_WIDTH"]):
                fig.add_trace(px.imshow(data[i*params["NETWORK_WIDTH"]+j])['data'][0], i+1, j+1)
        
        fig.write_html("results/figures/fig_"+str(config)+".html")

with Pool(8) as p:
    p.map(create_plots, range(50, 100))
        
#%%

df = pd.read_csv("/home/thomas/neuvisys-analysis/results/data2.csv")

features = {"Spikes": [], "MeanSize": [], "StdSize": [], "Vthresh": [], "Vreset": [], "DeltaVP": [], "DeltaVD": [], "TauLTP": [], "TauLTD": [], "TauM": [], "TauInhib": []}
for config in range(50, 100):
    with open("/home/thomas/neuvisys-analysis/results/config_files/conf_"+str(config)+".json") as file:
        params = json.load(file)
    features["Vthresh"].append(params["VTHRESH"])
    features["Vreset"].append(params["VRESET"])
    features["DeltaVP"].append(params["DELTA_VP"])
    features["DeltaVD"].append(params["DELTA_VD"])
    features["TauLTP"].append(params["TAU_LTP"])
    features["TauLTD"].append(params["TAU_LTD"])
    features["TauM"].append(params["TAU_M"])
    features["TauInhib"].append(params["TAU_INHIB"])
    
    nb_spikes = 0
    for entry in os.scandir("/home/thomas/neuvisys-analysis/results/weights/" + str(config) + "/"):
        if entry.path.endswith(".json"):
            with open(entry.path) as file:
                nb_spikes += json.load(file)["count_spike"]

    sizes = [os.path.getsize(file) for file in os.scandir("/home/thomas/neuvisys-analysis/results/metrics/"+str(config)+"/")]
    features["MeanSize"].append(np.mean(sizes))
    features["StdSize"].append(np.std(sizes))
    features["Spikes"].append(nb_spikes)

features = pd.DataFrame(features)
df_viz = pd.concat([df, features], axis=1)
df_viz["Dist_VP-VD"] = df_viz["DeltaVP"] - df_viz["DeltaVD"]
df_viz["Dist_LTP-LTD"] = df_viz["TauLTP"] - df_viz["TauLTD"]

features = (features - features.mean()) / features.std()
df = pd.concat([df, features], axis=1)

df["Dist_VP-VD"] = df["DeltaVP"] - df["DeltaVD"]
df["Dist_LTP-LTD"] = df["TauLTP"] - df["TauLTD"]

df["Dist_VP-LTP"] = df["DeltaVP"] - df["TauLTP"]
df["Dist_VD-LTD"] = df["DeltaVD"] - df["TauLTD"]

df["Dist_VP-LTD"] = df["DeltaVP"] - df["TauLTD"]
df["Dist_VD-LTP"] = df["DeltaVD"] - df["TauLTP"]

#%%

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
np.fill_diagonal(corr.values, 0)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# sns.pairplot(df[["Mark", "MeanSize", "StdSize"]])

#%%

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x = df.loc[:, features].values
y = df.loc[:, ["Mark"]].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[["Mark"]]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for target in targets:
    indicesToKeep = finalDf['Mark'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               s = 50)
ax.legend(targets)
ax.grid()

