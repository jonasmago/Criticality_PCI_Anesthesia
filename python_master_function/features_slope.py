#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy import signal
from utils import METHODS_chaos
import pandas as pd
# from mne.time_frequency import psd_multitaper, psd_welch
from fooof import FOOOF
import numpy as np
import argparse
import mne.io
import mne
import os
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat



# call:  python features_slope.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -lfrequ 1 -hfrequ 40

def features_slope (mne_epochs, lfreq, hfreq, fs=256, max_trials=30, bad_indices=None): 
    
    epochs = mne_epochs.get_data()
    nr_trials = min([len(epochs), max_trials])
    nr_channels =  epochs.shape[1]

    # search individual lowpass frequency
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [lfreq, hfreq]
    data_con = np.concatenate(epochs,axis = 1)
    # get psd of channels
    freqs, psds = signal.welch(data_con,fs,nperseg=5*1024)

    # Get average Slope interpoalted
    psds_interpoalted = psds.copy()
    psds_mean_interpolated =  np.mean(psds_interpoalted,axis = 0)
    fm = FOOOF()
    fm.fit(freqs, psds_mean_interpolated, freq_range)
    slope_id_interpoalted = -fm.aperiodic_params_[1]
    
    # Get average Slope not interpoalted
    psds[bad_indices,:] = np.nan
    psds_mean =  np.nanmean(psds,axis = 0)
    fm = FOOOF()
    fm.fit(freqs, psds_mean, freq_range)
    slope_id = -fm.aperiodic_params_[1]

    Slope_space_id = []
    # Get Space-resolved Slope
    for ch in range(len(psds_interpoalted)):
        fm = FOOOF()
        fm.fit(freqs, psds_interpoalted[ch,:], freq_range)
        slope_id = -fm.aperiodic_params_[1]
        Slope_space_id.append(slope_id)

    Slope_space_id = np.array(Slope_space_id)
    Slope_space_id_interpoalted = Slope_space_id.copy()
    Slope_space_id[bad_indices]=np.nan

    return slope_id, slope_id_interpoalted, Slope_space_id, Slope_space_id_interpoalted, np.array(psds_mean), np.array(psds_mean_interpolated), np.array(psds), np.array(psds_interpoalted), np.array(freqs), 
