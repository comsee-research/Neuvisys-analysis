#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:44:31 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def sphe_to_cart(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

r = 1
phi = np.linspace(0, np.pi, 25)
theta = np.linspace(0, 2*np.pi, 25)
mat = np.array(np.meshgrid(phi, theta)).T.reshape(-1, 2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*sphe_to_cart(r, mat[:, 0], mat[:, 1]), alpha=0.2)

n = 1000
k = 50
r = np.ones(n)
phi_dot = np.random.uniform(-1, 1, k)
theta_dot = np.random.uniform(-1, 1, k)
# theta_dot = np.sqrt(1 - phi_dot**2)
phi_dot = np.repeat(phi_dot, n / k)
theta_dot = np.repeat(theta_dot, n / k)
phi = np.zeros(n)
theta = np.zeros(n)
for i in range(n-1):
    phi[i+1] = phi[i] + phi_dot[i] / 100
    theta[i+1] = theta[i] + theta_dot[i] / 100

ax.scatter(*sphe_to_cart(r, phi, theta), c=np.linspace(0, 1, n), cmap="inferno")