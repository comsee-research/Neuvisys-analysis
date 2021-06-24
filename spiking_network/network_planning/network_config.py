# -*- coding: utf-8 -*-
from scipy.stats import truncnorm
import numpy as np


class NetworkConfig:
    def __init__(self):
        self.random_params = {
            "network_params": {
                "NbCameras": [1],
                "L1Width": [4],
                "L1Height": [4],
                "L1Depth": [100],
                "L1XAnchor": [[10, 148, 286]],
                "L1YAnchor": [[10, 105, 200]],
                "Neuron1Width": [10],
                "Neuron1Height": [10],
                "Neuron1Synapses": [1],
                "L2Width": [1],
                "L2Height": [1],
                "L2Depth": [16],
                "L2XAnchor": [[0, 4, 8]],
                "L2YAnchor": [[0, 4, 8]],
                "Neuron2Width": [4],
                "Neuron2Height": [4],
                "Neuron2Depth": [100],
                "L3Size": [0],
                "SharingType": ["patch"],
                "SaveData": [True],
            },
            "neuron_params": {
                "VTHRESH": truncnorm(a=(3 - 30) / 25, b=np.inf, loc=30, scale=25),
                "VRESET": truncnorm(a=-np.inf, b=(3 + 20) / 25, loc=-20, scale=25),
                "TRACKING": ["partial"],
                "TAU_SRA": truncnorm(
                    a=(1000 - 100000) / 50000, b=np.inf, loc=100000, scale=50000
                ),
                "TAU_RP": truncnorm(
                    a=(100 - 20000) / 25000, b=np.inf, loc=20000, scale=25000
                ),
                "TAU_M": truncnorm(
                    a=(100 - 18000) / 25000, b=np.inf, loc=18000, scale=25000
                ),
                "TAU_LTP": truncnorm(
                    a=(100 - 7000) / 25000, b=np.inf, loc=7000, scale=25000
                ),
                "TAU_LTD": truncnorm(
                    a=(100 - 14000) / 25000, b=np.inf, loc=14000, scale=25000
                ),
                "TARGET_SPIKE_RATE": truncnorm(
                    a=(0.01 - 0.75) / 1, b=np.inf, loc=0.75, scale=1
                ),
                "SYNAPSE_DELAY": [0],
                "STDP_LEARNING": [True],
                "NORM_FACTOR": truncnorm(a=(0 - 4) / 2, b=np.inf, loc=4, scale=2),
                "MIN_THRESH": [4],
                "ETA_LTP": truncnorm(
                    a=(0.0001 - 0.0077) / 0.008, b=np.inf, loc=0.0077, scale=0.008
                ),
                "ETA_LTD": truncnorm(
                    a=-np.inf, b=(0.0001 + 0.0021) / 0.002, loc=-0.0021, scale=0.002
                ),
                "ETA_SRA": truncnorm(a=(0.1 - 0.6) / 100, b=np.inf, loc=0.6, scale=100),
                "ETA_TA": truncnorm(a=(0.1 - 1) / 10, b=np.inf, loc=1, scale=10),
                "ETA_RP": truncnorm(a=(0.1 - 1) / 10, b=np.inf, loc=1, scale=10),
                "ETA_INH": truncnorm(a=(0 - 20) / 5, b=np.inf, loc=20, scale=5),
                "DECAY_FACTOR": [0],
            },
            "pooling_neuron_params": {
                "VTHRESH": [3],
                "VRESET": [-20],
                "TRACKING": ["partial"],
                "TAU_M": [20000],
                "TAU_LTP": [20000],
                "TAU_LTD": [20000],
                "STDP_LEARNING": [True],
                "NORM_FACTOR": [10],
                "ETA_LTP": [0.2],
                "ETA_LTD": [0.2],
                "ETA_INH": [25],
                "ETA_RP": [1],
                "TAU_RP": [20000],
                "DECAY_FACTOR": [0],
            },
            "motor_neuron_params": {
                "VTHRESH": [3],
                "VRESET": [-20],
                "TRACKING": ["partial"],
                "TAU_M": [20000],
                "ETA_INH": [25],
            },
        }

        self.params = {
            "network_params": {
                "NbCameras": [1],
                "L1Width": [4],
                "L1Height": [4],
                "L1Depth": [100],
                "L1XAnchor": [[10, 148, 286]],
                "L1YAnchor": [[10, 105, 200]],
                "Neuron1Width": [10],
                "Neuron1Height": [10],
                "Neuron1Synapses": [1],
                "L2Width": [1],
                "L2Height": [1],
                "L2Depth": [16],
                "L2XAnchor": [[0, 4, 8]],
                "L2YAnchor": [[0, 4, 8]],
                "Neuron2Width": [4],
                "Neuron2Height": [4],
                "Neuron2Depth": [100],
                "L3Size": [0],
                "SharingType": ["patch"],
                "SaveData": [True],
            },
            "neuron_params": {
                "VTHRESH": [30],
                "VRESET": [-20],
                "TRACKING": ["partial"],
                "TAU_SRA": [100000],
                "TAU_RP": [20000],
                "TAU_M": [18000],
                "TAU_LTP": [7000],
                "TAU_LTD": [14000],
                "TARGET_SPIKE_RATE": [0.75],
                "SYNAPSE_DELAY": [0],
                "STDP_LEARNING": [True],
                "NORM_FACTOR": [4],
                "MIN_THRESH": [4],
                "ETA_LTP": [0.0077],
                "ETA_LTD": [-0.0021],
                "ETA_SRA": [0.6],
                "ETA_TA": [1],
                "ETA_RP": [1],
                "ETA_INH": [20],
                "DECAY_FACTOR": [0],
            },
            "pooling_neuron_params": {
                "VTHRESH": [3],
                "VRESET": [-20],
                "TRACKING": ["partial"],
                "TAU_M": [20000],
                "TAU_LTP": [20000],
                "TAU_LTD": [20000],
                "STDP_LEARNING": [True],
                "NORM_FACTOR": [10],
                "ETA_LTP": [0.2],
                "ETA_LTD": [0.2],
                "ETA_INH": [25],
                "ETA_RP": [1],
                "TAU_RP": [20000],
                "DECAY_FACTOR": [0],
            },
            "motor_neuron_params": {
                "VTHRESH": [3],
                "VRESET": [-20],
                "TRACKING": ["partial"],
                "TAU_M": [20000],
                "ETA_INH": [25],
            },
        }
