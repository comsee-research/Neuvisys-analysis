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

g, d = centroids(spinet)
gd = g - d

fig, axes = plt.subplots(4, 5, sharex=True, sharey=True)
for i in range(5):
    for j in range(4):
        # axes[j, i].suptitle("region: "+str(1+i)+", "+str(1+j) + "\n" + "mean (x, y):" + str(np.mean(gd[spinet.layout1[i, j, 0]:spinet.layout1[i+2, j+2, 0]], axis=0)) + "\n" + "std  (x, y):" + str(np.std(gd[spinet.layout1[i, j, 0]:spinet.layout1[i+2, j+2, 0]], axis=0)))
        # axes[j, i].set_xlim([-5, 5])
        # axes[j, i].set_ylim([0, 250])
        axes[j, i].hist(gd[spinet.layout1[i*3, j*3, 0]:spinet.layout1[i*3+2, j*3+2, 0], 0])
        
fig, axes = plt.subplots(4, 5, sharex=True, sharey=True)
for i in range(5):
    for j in range(4):
        # axes[j, i].suptitle("region: "+str(1+i)+", "+str(1+j) + "\n" + "mean (x, y):" + str(np.mean(gd[spinet.layout1[i, j, 0]:spinet.layout1[i+2, j+2, 0]], axis=0)) + "\n" + "std  (x, y):" + str(np.std(gd[spinet.layout1[i, j, 0]:spinet.layout1[i+2, j+2, 0]], axis=0)))
        # axes[j, i].set_xlim([-5, 5])
        # axes[j, i].set_ylim([0, 250])
        axes[j, i].hist(gd[spinet.layout1[i*3, j*3, 0]:spinet.layout1[i*3+2, j*3+2, 0], 1])
        
#%%

xmu = gabor_params_l[0][0] - gabor_params_r[0][0]
ymu = gabor_params_l[0][1] - gabor_params_r[0][1]
xsigma = gabor_params_l[1][0] - gabor_params_r[1][0]
ysigma = gabor_params_l[1][1] - gabor_params_r[1][1]
lambd = gabor_params_l[2] - gabor_params_r[2]
theta = gabor_params_l[4] - gabor_params_r[4]
error = (gabor_params_l[5] + gabor_params_r[5]) / 2

disp = (gabor_params_l[3] - gabor_params_r[3]) / (2 * np.pi * ((gabor_params_l[2] + gabor_params_r[2]) / 2) * np.cos(((gabor_params_l[4] + gabor_params_r[4]) / 2)))

epsi_xmu = 80
epsi_ymu = 60
epsi_xsigma = 20
epsi_ysigma = 60
epsi_lambd = 3
epsi_theta = 0.5
epsi_error = 5

id_disp = np.where((np.abs(xmu) < epsi_xmu) & (np.abs(ymu) < epsi_ymu) & (np.abs(xsigma) < epsi_xsigma) & (np.abs(ysigma) < epsi_ysigma) & (np.abs(lambd) < epsi_lambd) & (np.abs(theta) < epsi_theta) & (error < epsi_error))[1]
disp_f = disp[(np.abs(xmu) < epsi_xmu) & (np.abs(ymu) < epsi_ymu) & (np.abs(xsigma) < epsi_xsigma) & (np.abs(ysigma) < epsi_ysigma) & (np.abs(lambd) < epsi_lambd) & (np.abs(theta) < epsi_theta) & (error < epsi_error)]
final_disp = np.vstack((id_disp, disp_f))

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

