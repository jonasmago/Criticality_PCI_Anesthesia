#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy import signal
import pandas as pd
# from mne.time_frequency import psd_multitaper, psd_welch
from fooof import FOOOF
import numpy as np
import argparse
import mne.io
import mne
import os
import neurokit2 as nk
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat



# call:  python features_DFA.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -lfrequ 8 -hfrequ 14

def get_channel_hurst(ch_data,sfreq):
    scale = nk.expspace(1*sfreq, 20*sfreq, 40, base=2).astype(np.int64)
    scale = nk.expspace(1*sfreq, 3*sfreq, 5, base=2).astype(np.int64)

    analytic_signal = hilbert(ch_data)
    amplitude_envelope = np.abs(analytic_signal)

    hurst_fh, _ = nk.fractal_hurst(amplitude_envelope, scale=scale, show=False)
    hurst_dfa, _ = nk.fractal_dfa(amplitude_envelope, scale=scale, show=False)
    print ('check3')
    print ('done one round of hurst')
    return  hurst_fh, hurst_dfa


def features_DFA(raw, lfreq, hfreq, fs=256, max_s=200, bad_indices=None):
        data = raw.get_data()
        nr_channels =  data.shape[0]

        # cut data and only use first 200s or less
        sig_length = min(data.shape[1]/fs , 200)
        cut = int(sig_length*fs)
        data = data[:,:cut]

        data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=lfreq, h_freq=hfreq,verbose=False)

        input = []
        for ch in range(nr_channels):
            input.append((data_filt[ch,:],fs))
            get_channel_hurst(data_filt[ch,:],fs)


        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(get_channel_hurst,input)
        pool.close()
        
        results = np.array(results)
        results_interpolated = results.copy()
        results[bad_indices,:] = np.nan

        HURST_FH = np.nanmean(results[:,0])
        HURST_DFA = np.nanmean((results)[:,1])
        HURST_FH_interpolated = np.mean(results_interpolated[:,0])
        HURST_DFA_interpolated = np.mean((results_interpolated)[:,1])

        return HURST_FH, HURST_DFA, results, HURST_FH_interpolated, HURST_DFA_interpolated, results_interpolated
