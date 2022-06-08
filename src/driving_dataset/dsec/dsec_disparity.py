#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:13:06 2022

@author: thomas
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

path = "/home/thomas/Bureau/interlaken_00_c_disparity_event"
images_path = natsorted(os.listdir(path))

sequence = []
for image_path in images_path:
    with Image.open("/home/thomas/Bureau/interlaken_00_c_disparity_event/" + image_path) as file:
        image = np.asanyarray(file) / 256
        image = image / 6
        sequence.append(image)
sequence = np.array(sequence)

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.show()

plt.figure()
plt.hist(sequence[sequence != 0].flatten(), bins=100)
plt.show()
