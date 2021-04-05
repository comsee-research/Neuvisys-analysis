#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:19:28 2021

@author: thomas
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def stereo_matching(folder, xs, ys, range_imgs):
    mat = np.zeros((346, 260))
    ind = 1
    for x in xs:
        for y in ys:
            mat[x : x + 30, y : y + 30] = ind
            ind += 1

    vec = {}
    for i in range(21):
        vec[i] = []

    for i in np.arange(0, 824):
        lframe = cv.imread(folder + "img" + str(i) + "_left.jpg")
        rframe = cv.imread(folder + "img" + str(i) + "_right.jpg")

        orb = cv.ORB_create(nfeatures=1000)

        kp_left, ds_left = orb.detectAndCompute(lframe, None)
        kp_right, ds_right = orb.detectAndCompute(rframe, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ds_left, ds_right)

        matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            lp = kp_left[match.queryIdx].pt
            rp = kp_right[match.trainIdx].pt

            x_shift = lp[0] - rp[0]
            y_shift = lp[1] - rp[1]
            # print("{:.1f}, {:.1f}".format(*lp), "|", "{:.1f}, {:.1f}".format(*rp), "->", "{:.2f}".format(x_shift), "|", "{:.2f}".format(y_shift))

            if np.abs(x_shift) < 20 and np.abs(y_shift) < 20:
                vec[mat[int(np.round((lp[0]))), int(np.round(lp[1]))]].append(
                    [x_shift, y_shift]
                )

            # imgmatching = cv.drawMatches(lframe, kp_left, rframe, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(imgmatching)

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    fin = np.zeros((len(ys), len(xs), 2))
    nb_fin = np.zeros((len(ys), len(xs)))
    ind = 1
    for i in range(len(xs)):
        for j in range(len(ys)):
            axes[j, i].set_title("nb : " + str(np.array(vec[ind])[:, 0].shape[0]))
            axes[j, i].hist(
                np.array(vec[ind])[:, 0], np.arange(-25.5, 26.5), density=True
            )
            fin[j, i] = np.mean(vec[ind], axis=0)
            nb_fin[j, i] = len(vec[ind])
            ind += 1
