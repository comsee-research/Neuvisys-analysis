#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:47:24 2020

@author: alphat
"""

#%%

path = "/media/alphat/SSD Games/Thesis/configuration/network_"

a = []
for i in range(8):
    with open(path+str(i)+"/configs/pooling_neuron_config.json") as file:
        a.append(json.load(file))

df = pd.DataFrame(a)


#%%

network_path = ""
spinet = SpikingNetwork(network_path)

        
#%% Spike rate

network_path = "/home/alphat/neuvisys-dv/configuration/network/"
time = np.max(spinet.sspikes)

srates = np.count_nonzero(spinet.sspikes, axis=1) / (time * 1e-6)
print("mean:", np.mean(srates))
print("std:", np.std(srates))


#%%
from fpdf import FPDF
from natsort import natsorted
import random

row = 6
col = 6
im_size = (20, 20)
im_size_pad = (22, 22)

pdf = FPDF("P", "mm", (col*im_size_pad[0], row*im_size_pad[1]))
pdf.add_page()

path = "/home/alphat/neuvisys-dv/configuration/NETWORKS/complex_selectivity/figures/complex_directions/"

images = natsorted(os.listdir(path))#[::-1]
random.shuffle(images)

count = 0
for i in range(row):
    for j in range(col):
        pdf.image(path+images[count], x=j*im_size_pad[0], y=i*im_size_pad[1], w=im_size[0], h=im_size[1])
        count += 1
        
pdf.output("/home/alphat/Desktop/images/directions.pdf", "F")

#%%

rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
dir_vec, ori_vec = complex_cells_directions(spinet, rotations)

angles = np.pi * rotations / 180

dirs = []
dis = []
for i in range(144):
    dirs.append(direction_norm_length(spinet.directions[:, i], angles))
    dis.append(direction_selectivity(spinet.directions[:, i]))
oris = []
ois = []
for i in range(144):
    oris.append(orientation_norm_length(spinet.orientations[:, i], angles[0:8]))
    ois.append(orientation_selectivity(spinet.orientations[:, i]))
    
#%%

fig, ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

color1 = dict(color="#2C363F")
color2 = dict(color="#9E7B9B")

ax.set_ylabel("normalized vector length")
ax.boxplot(np.abs(des), positions=[0], labels=["direction exponential"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(oes), positions=[1], labels=["orientation exponential"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dls), positions=[2], labels=["direction linear"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(ols), positions=[3], labels=["orientation linear"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dsr), positions=[4], labels=["direction step left"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(osr), positions=[5], labels=["orientation step left"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dsl), positions=[6], labels=["direction step right"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(osl), positions=[7], labels=["orientation step right"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(dss), positions=[8], labels=["direction step sym"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
ax.boxplot(np.abs(oss), positions=[9], labels=["orientation step sym"], boxprops=color1, medianprops=color2, whiskerprops=color1, capprops=color1, flierprops=dict(markeredgecolor=color1["color"]))
plt.show()


#%%

eta_ltp = 0.2
eta_ltd = -0.2
tau_ltp = 20
tau_ltd = 20

t = np.linspace(-80, 80, 10000)

step = np.zeros(10000)
step[(t > -tau_ltd) & (t <= 0)] = -eta_ltd
step[(t >= 0) & (t < tau_ltd)] = eta_ltp

lin = np.zeros(10000)
lin[(t > -tau_ltd) & (t <= 0)] = -eta_ltd * -t[(t > -tau_ltd) & (t <= 0)][::-1] / np.max(-t[(t > -tau_ltd) & (t <= 0)][::-1])
lin[(t >= 0) & (t < tau_ltd)] = eta_ltp * t[(t >= 0) & (t < tau_ltd)][::-1] / np.max(t[(t >= 0) & (t < tau_ltd)][::-1])

exp = np.concatenate((eta_ltd * np.exp(t[t < 0] / tau_ltd), eta_ltp * np.exp(-t[t >= 0] / tau_ltp)))


fig, axs = plt.subplots(3, 1)

for ax in axs.flat:
    ax.grid(alpha=.2, linestyle='--')
    ax.axhline(0, alpha=0.3, linestyle='-', color='k')
    ax.axvline(0, alpha=0.3, linestyle='-', color='k')
    ax.xlim = [-0.5, 0.5]
    ax.set_ylabel("Synaptic change (mV)")
    ax.label_outer()

axs[1].set_xlabel("Spike timing (ms)")

axs[0].plot(t, step, color='k')
axs[0].set_title("Step")
axs[1].plot(t, lin, color='k')
axs[1].set_title("Linear")
axs[2].plot(t, exp, color='k')
axs[2].set_title("Exponential")


#%%

block_size = 3
min_disp = 0
num_disp = 16

stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=block_size,
                              P1=8*1*block_size**2,
                              P2=32*1*block_size**2,
                              disp12MaxDiff=1,
                              preFilterCap=0,
                              uniquenessRatio=10,
                              speckleWindowSize=50,
                              speckleRange=1,
                              mode=cv.StereoSGBM_MODE_HH)
for i in range(rect_frames.shape[1]):
    disparity = stereo.compute(rect_frames[0, i], rect_frames[1, i])
    cv.imwrite("/home/alphat/Desktop/disp/"+str(i)+".jpg", disparity)
    # plt.figure()
    # plt.imshow(disparity, 'gray')

