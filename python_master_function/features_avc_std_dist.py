#!/usr/bin/env python
# from mne.time_frequency import psd_multitaper, psd_welch
from scipy.signal import hilbert
from scipy.signal import welch
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
from fooof import FOOOF
import edgeofpy as eop
from scipy import stats
import pandas as pd
import numpy as np
import argparse
import pickle
import mne.io
import mne
import os




# call:  python features_avc_std_dist.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt
def features_avc_std_dist (raw, max_s=300, fs=256):

    FIL_FREQ = (0.5, 40) # bandpass frequencies
    THRESH_TYPE = 'both' # Fosque22: 'both'



    hist_x = []
    hist_y = []

    hist_x_raw = []
    hist_y_raw = []

    data = raw.get_data()
    nr_channels =  data.shape[0]
    sig_length = min(data.shape[1]/fs , max_s)
    cut = int(sig_length*fs)
    data = data[:,:cut]

    data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=FIL_FREQ[0], h_freq=FIL_FREQ[1],verbose=False)

    # FIND DEVIATIONS FOR RELATIVE STD (Per recording)
    mean_per_chan = np.mean(data_filt, axis=1, keepdims=True)
    std_per_chan = np.std(data_filt, axis=1, keepdims=True)

    # Z-score all data
    data_z = (data_filt-mean_per_chan)/std_per_chan

    scale = np.linspace(-10,10,100)
    part_hist_y, part_hist_x = np.histogram(data_z, bins = scale)
    
    scale = np.linspace(-0.0001,0.0001,100)
    part_hist_y_raw, part_hist_x_raw = np.histogram(data_filt, bins = scale)

    return part_hist_x, part_hist_y, part_hist_x_raw, part_hist_y_raw

