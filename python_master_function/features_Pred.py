#!/usr/bin/env python
import pandas as pd
from antropy import lziv_complexity
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import argparse
import mne.io
import random
import mne
import os
import multiprocessing as mp
import neurokit2 as nk
from scipy.fft import fft, ifft
from scipy.signal import hilbert
from scipy.fftpack import rfft, irfft

# call:   python features_Pred.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -lfrequ 0.5 -hfrequ 40


# Ent_mean is currnetly not working!!!!

def get_pred(trial, epochs):
    Lyaps = []
    Dims = []
    Ent = []
    LZC = []
    KDF = []

    nr_channels =  epochs.shape[1]
    trial_data = epochs[trial]

    for ch in range(nr_channels):
        print (ch)
        channel_data = trial_data[ch]
        lle, _ = nk.complexity_lyapunov(channel_data, method="rosenstein1993", show=False)
        Lyaps.append(lle)

        dims, _ = nk.complexity_dimension(channel_data)
        Dims.append(dims)

        ent, _ = nk.entropy_multiscale(channel_data, show=False, dimension=dims)
        Ent.append(ent)

        lzc, _ = nk.complexity_lempelziv(channel_data, show=False)
        LZC.append(lzc)

        kdf, _ = nk.fractal_katz(channel_data)
        KDF.append(kdf)

    print('Done Trial {}'.format(str(trial)))

    return Lyaps, Dims, Ent, LZC, KDF


def features_Pred (mne_epochs, lfreq, hfreq, fs=256, max_trials=30, bad_indices = None):
    
    # epochs_res = mne_epochs.resample(250)
    epochs_res = mne_epochs
    epochs_filt = epochs_res.filter(lfreq, hfreq, verbose = False)

    # if data is too long only use the first 3 min of data
    nr_trials = min([len(epochs_filt),max_trials])
    epochs = epochs_filt.get_data()
    nr_channels =  epochs.shape[1]

    #################################
    #    Calculate LZC             #
    #################################

    pool = mp.Pool(mp.cpu_count())

    # loop over every time segment
    input = []
    for trial in range(nr_trials):
        input.append((trial,epochs))

    results = pool.starmap(get_pred,input)

    results = np.array(results) 
    results[:, :, bad_indices] = np.nan

    results_int = np.array(results) 

    Lyaps_max = (np.nanmean(results[:,0,:],axis = (0,1)))
    Dims_mean = (np.nanmean(results[:,1,:],axis = (0,1)))
    Ent_mean = (np.nanmean(results[:,2,:],axis = (0,1)))
    LZC_mean = (np.nanmean(results[:,3,:],axis = (0,1)))
    KDF_mean = (np.nanmean(results[:,4,:],axis = (0,1)))

    Lyaps_max_int = (np.nanmean(results_int[:,0,:],axis = (0,1)))
    Dims_mean_int = (np.nanmean(results_int[:,1,:],axis = (0,1)))
    Ent_mean_int = (np.nanmean(results_int[:,2,:],axis = (0,1)))
    LZC_mean_int = (np.nanmean(results_int[:,3,:],axis = (0,1)))
    KDF_mean_int = (np.nanmean(results_int[:,4,:],axis = (0,1)))

    pool.close()
    
    return Lyaps_max, Dims_mean, Ent_mean, LZC_mean, KDF_mean, Lyaps_max_int, Dims_mean_int, Ent_mean_int, LZC_mean_int, KDF_mean_int, results, results_int 
