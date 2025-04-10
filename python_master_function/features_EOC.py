#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy.signal import butter, lfilter,firls
from scipy import signal
from utils import METHODS_chaos
import pandas as pd
from scipy import stats
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from fooof import FOOOF
import numpy as np
import argparse
import mne.io
import mne
import os
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat



# call:  python features_EOC.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -ktype flex

def fixed_chaos(trial, epochs, lpfrequency):
    K_ch = []
    hfreq = []
    failed = []

    # select trial from epoch
    data_trial = epochs[trial]
    fs = 256
    samples = data_trial.shape[1]
    nr_channels =  epochs.shape[1]

    for ch in range(nr_channels):
        # select channel data
        data_ch = data_trial[ch,:]
        ch_filt = mne.filter.filter_data(data_ch, sfreq=fs, l_freq=0.5, h_freq=lpfrequency,verbose=False)
        K_tmp = METHODS_chaos.chaos_pipeline(ch_filt)
        K_ch.append(K_tmp)
        hfreq.append(lpfrequency)
        if type(K_tmp) != np.nan:
            failed.append(0)
        else:
            failed.append(1)

    print(f'Done Trial {str(trial)} Fixed {str(lpfrequency)} Hz')

    return K_ch, hfreq, failed


def filter_and_chaos(trial, epochs):
    K_ch = []
    hfreq = []
    failed = []

    # select trial from epoch
    data_trial = epochs[trial]
    fs = 256
    samples = data_trial.shape[1]

    nr_channels =  epochs.shape[1]
    for ch in range(nr_channels):
        # select channel data
        data_ch = data_trial[ch,:]
        # do FOOOF to find lowst frequency peak
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [1, 6]
        # get psd of channels
        freqs, psd_ch = signal.welch(data_ch,fs,nperseg=samples)

        fm.fit(freqs, psd_ch, freq_range)

        if fm.peak_params_.shape[0] == 0:
            #no peak found, output nan
            failed.append(1)
            hfreq.append( np.nan )
            K_ch.append( np.nan )
        elif fm.peak_params_.shape[0] >= 1:
            failed.append(0)
            # select lowest frequency peak
            peak = fm.peak_params_[np.where(fm.peak_params_[:,0] == np.min(fm.peak_params_[:,0]))[0][0]]
            hfreq_tmp = peak[0] + 0.5*peak[2] #higher edge of lowest frequency
            #Filter data at chosen lowest frequency
            ch_filt = mne.filter.filter_data(data_ch, sfreq=fs, l_freq=0.5, h_freq=hfreq_tmp,verbose=False)
            K_ch.append(METHODS_chaos.chaos_pipeline(ch_filt))
            hfreq.append(hfreq_tmp)
    print('Done Trial {}'.format(str(trial)))

    return K_ch, hfreq, failed


def features_EOC(mne_epochs, k_type='flex', hfrequ=None, max_trials=30, bad_indices=None, good_indices=None):

    # output
    Freq = []
    Nopeak = []
    K_median = []
    K_space = []

    mne_epochs.pick_types(eeg=True)
    mne_epochs.drop_channels(mne_epochs.info['bads'])
    epochs = mne_epochs.get_data()
    fs = 256
    samples = epochs[0].shape[1]

    # if data is too long only use the first 3 min of data
    nr_trials = min([len(epochs),max_trials]);
    nr_channels =  epochs.shape[1]

    # search individual lowpass frequency
    if k_type == 'indflex':
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [1, 6]
        data_con = np.concatenate(epochs,axis = 1)
        # get psd of channels
        freqs, psds = signal.welch(data_con,fs,nperseg=5*1024)
        psds =  np.mean(psds,axis = 0)
        fm.fit(freqs, psds, freq_range)
        if fm.peak_params_.shape[0] == 0:
            hfrequ = 4
        if fm.peak_params_.shape[0] >= 1:
            # select lowest frequency peak
            peak = fm.peak_params_[np.where(fm.peak_params_[:,0] == np.min(fm.peak_params_[:,0]))[0][0]]
            hfrequ = peak[0] + 0.5*peak[2] #higher edge of lowest frequency


    #################################
    #    Calculate 01chaos test     #
    #################################

    pool = mp.Pool(mp.cpu_count())
    # loop over every time segment

    input = []
    for trial in range(nr_trials):
        if k_type == 'indflex':
            input.append((trial,epochs, np.float64(hfrequ)))
        if k_type == 'flex':
            input.append((trial,epochs))
            #filter_and_chaos(trial,epochs) # comment out
        elif ktype == 'fixed':
            input.append((trial,epochs, np.float64(hfrequ)))

        #filter_and_chaos(trial,epochs)
        #fixed_chaos(trial,epochs, np.float(hfrequ))

    #get results for chaos test
    if k_type == 'flex':
        results = pool.starmap(filter_and_chaos,input)
    else:
        results = pool.starmap(fixed_chaos,input)

    results_array_interpolated = np.array(results)
    results_array = results_array_interpolated.copy()
    results_array[:, :, bad_indices] = np.nan



    K_median = np.nanmedian(results_array[:,0,:])
    K_median_interpolated = np.nanmedian(results_array_interpolated[:,0,:])
    Freq = np.nanmean(results_array[:,1,:])
    Nopeak = np.sum(results_array[:,2,:])/(nr_trials*nr_channels)


    pool.close()
    return K_median, K_median_interpolated, Freq, Nopeak, results_array, results_array_interpolated

