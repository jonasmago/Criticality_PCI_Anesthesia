#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy.signal import butter, lfilter,firls
from scipy import signal
import sys
import os
# setting path
#sys.path.append('../')
from utils import METHODS_EOS
import pandas as pd
from fooof import FOOOF
import numpy as np
import argparse
import mne.io
import mne
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat

# call:  python features_EOS.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -minfreq 8 -maxfreq 14


def calculate_values(trial, epochs):

    data_tr = epochs[trial].get_data()[0]

    #PLE = Methods_EOS.ple(data_tr, m = 5, tau = 2)
    #PLI = Methods_EOS.pli(data_tr)
    PCF, OR_mean, orph_vector_tr, orpa_vector_tr = METHODS_EOS.pcf(data_tr)
    print(f'done Trial {str(trial)}')

    #return PLI, PLE, PCF, OR_mean, OR_var
    return PCF, OR_mean

def features_EOS (mne_epochs, minfreq, maxfreq, fs, max_trials=30):

        # prepare data
        # epochs_res = mne_epochs.resample(250)
        epochs_res = mne_epochs
        epochs_filt = epochs_res.filter(minfreq, maxfreq, verbose = False)

        # if data is too long only use the first 3 min of data
        nr_trials = min([len(epochs_filt),max_trials]);
        nr_channels =  epochs_filt.info['nchan']

        ###############################################
        #    Calculate Pair Correlation Function     #
        ###############################################

        pool = mp.Pool(mp.cpu_count())
        # loop over every time segment

        # prepare input for parallel function
        input = []
        for trial in range(nr_trials):
            input.append((trial,epochs_filt))

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(calculate_values,input)
        pool.close()

        

        PCF_mean = np.median(pd.DataFrame(results)[0])
        OR_mean = np.median(pd.DataFrame(results)[1])

        return PCF_mean, OR_mean, results

