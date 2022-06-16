#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:19:28 2021

@author: thomas
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import rosbag
import seaborn as sns
from dv import AedatFile
from PIL import Image, ImageDraw
import subprocess
import os


def load_frames(path):
    with AedatFile(path) as f:
        try:
            lframes = []
            rframes = []
            for l, r in zip(f["frames"], f["frames_1"]):
                lframes.append(l.image)
                rframes.append(r.image)
            return np.array(lframes), np.array(rframes)
        except:
            frames = []
            for fr in f["frames"]:
                frames.append(fr.image)
            return np.array(frames)


def shift(arr, num, axis, fill_value=0):
    arr = np.roll(arr, num, axis=axis)
    if num < 0:
        if axis == 1:
            arr[:, num:] = fill_value
        if axis == 2:
            arr[:, :, num:] = fill_value
    elif num > 0:
        if axis == 1:
            arr[:, :num] = fill_value
        if axis == 2:
            arr[:, :, :num] = fill_value
    return arr


def extract_frames_video(video_paths: []):
    for i, video_path in enumerate(video_paths):
        folder = video_path.rsplit("/", 1)[0] + "/" + str(i)
        os.mkdir(folder)
        subprocess.run(["ffmpeg", "-i", video_path, folder + "/" + "%03d.bmp"])


def rectify_frames(frames, lx, ly, rx, ry):
    rect_frames = np.array(frames).copy()
    rect_frames[0] = shift(rect_frames[0], ly, 1)
    rect_frames[0] = shift(rect_frames[0], lx, 2)
    rect_frames[1] = shift(rect_frames[1], ry, 1)
    rect_frames[1] = shift(rect_frames[1], rx, 2)

    return rect_frames


def write_frames(dest, frames, rec):
    for i in range(frames.shape[1]):
        lim = Image.fromarray(np.squeeze(frames[0][i], axis=2))
        rim = Image.fromarray(np.squeeze(frames[1][i], axis=2))
        ldraw = ImageDraw.Draw(lim)
        rdraw = ImageDraw.Draw(rim)
        for x in rec[0]:
            for y in rec[1]:
                ldraw.rectangle([x, y, x + 31, y + 31], outline=255)
                rdraw.rectangle([x, y, x + 31, y + 31], outline=255)
        lim.save(dest + "img" + str(i) + "_left.jpg")
        rim.save(dest + "img" + str(i) + "_right.jpg")


def load_depth_images_rosbag(bag_file):
    bag = rosbag.Bag(bag_file)
    dt = np.dtype("<f4")

    xs = [10, 148, 286]
    ys = [10, 105, 200]
    mat = [[[], [], []], [[], [], []], [[], [], []]]

    for topic, msg, t in bag.read_messages(topics=["/davis/left/depth_image_raw"]):
        # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        im = np.frombuffer(msg.data, dtype=dt).reshape(260, 346)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                mat[i][j].append(im[y: y + 40, x: x + 40].flatten())
        # plt.imshow(im)
        # plt.show()

    mat = np.array(mat)
    mat = mat.reshape(mat.shape[:-2] + (-1,))


def depth_estimation_per_region(xs, ys, mat):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
    # fig.suptitle("Ground Truth Depth estimation per region", fontsize=30)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if j != 2:
                sns.histplot(
                    mat[i, j][mat[i, j] < 100],
                    ax=axes[j, i],
                    stat="density",
                    color="#2C363F",
                )

    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=15)
        ax.set_ylabel("Density", fontsize=26)
        ax.set_xticks(np.arange(0, 110, 10))
    axes[1, 1].set_xlabel("Depth (m)", fontsize=26)

    plt.savefig("/home/thomas/Desktop/images/gt", bbox_inches="tight")


def stereo_matching(folder, xs, ys, range_imgs):
    mat = np.zeros((346, 260))
    ind = 1
    for x in xs:
        for y in ys:
            mat[x: x + 30, y: y + 30] = ind
            ind += 1

    vec = {}
    for i in range(21):
        vec[i] = []

    for i in np.arange(100, 824):
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
            #print("{:.1f}, {:.1f}".format(*lp), "|", "{:.1f}, {:.1f}".format(*rp), "->", "{:.2f}".format(x_shift), "|", "{:.2f}".format(y_shift))

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
