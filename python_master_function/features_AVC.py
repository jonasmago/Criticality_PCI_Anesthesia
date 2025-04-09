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
import powerlaw
import argparse
import pickle
import mne.io
import mne
import os

np.int = int  # Patch deprecated alias before anything else uses it


# call:  python features_AVC.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -bin_treshold 1.5 -max_iei 0.004

def features_AVC (raw, bin_threshold, max_iei, fs=256, max_s=200):


    FIL_FREQ = (0.5, 40) # bandpass frequencies
    THRESH_TYPE = 'both' # Fosque22: 'both'

    GAMMA_EXPONENT_RANGE = (0, 2)
    LATTICE_SEARCH_STEP = 0.1

    # read out arguments
    BIN_THRESHOLD = float(bin_threshold)
    MAX_IEI = float(max_iei)
    BRANCHING_RATIO_TIME_BIN = float(max_iei)

    # output
    out = {'mean_iei':[],
           'tau':[],
           'tau_dist':[],
           'tau_dist_TR':[],
           'alpha':[],
           'alpha_dist':[],
           'alpha_dist_TR':[],
           'third':[],
           'dcc_cn':[],
           'avl_br':[],
           'br':[],
           'rep_dissimilarity_avg':[],
           'rep_size':[],
           'fano':[],
           'chi_test':[],
           'chi_notest':[],
           'sig_length':[],
           'len_avls':[],
           'data_mean':[],
           'data_std':[]
           }



    data = raw.get_data()
    nr_channels =  data.shape[0]
    sig_length = min(data.shape[1]/fs , max_s)
    cut = int(sig_length*fs)
    data = data[:,:cut]

    data_mean = np.mean(np.abs(data))
    data_std = np.std(data)

    data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=FIL_FREQ[0], h_freq=FIL_FREQ[1],verbose=False)

    events_by_chan = eop.binarized_events(data_filt, threshold=BIN_THRESHOLD,
                                thresh_type=THRESH_TYPE, null_value=0)
    events_one_chan = np.sum(events_by_chan, axis=0)


    #################################
    #    Avalanches                 #
    #################################

    # Detect avalanches
    #breakpoint()
    avls, _, _, mean_iei = eop.detect_avalanches(events_by_chan, fs,
                                                    max_iei=MAX_IEI,
                                                    threshold=BIN_THRESHOLD,
                                                    thresh_type=THRESH_TYPE)

    sizes = [x['size'] for x in avls]
    dur_bin = [x['dur_bin'] for x in avls]
    dur_sec = [x['dur_sec'] for x in avls if not np.isnan(x['dur_sec'])]  # skip NaNs
    # dur_sec = [x['dur_sec'] for x in avls]
    len_avls = len(avls)

    # Skip if no valid avalanches
    if len(sizes) == 0 or len(dur_sec) == 0:
        for name in out.keys():
            out[name].append(np.nan)
        return out, avls

    #################################
    #    TAU                 #
    #################################
    # Estimate fit and extract exponents with min and max of data
    size_fit = eop.fit_powerlaw(sizes, xmin=1, discrete = True, xmax = None)
    tau = size_fit['power_law_exp']
    tau_dist = size_fit['best_fit']
    tau_dist_TR = size_fit['T_R_sum']


    #################################
    #    ALPHA                     #
    #################################

    #dur_bin_fit = eop.fit_powerlaw(dur_bin, discrete = True)
    #alpha_bin = dur_bin_fit['power_law_exp']

    print (f'number of elements: {len(dur_sec)}')
    dur_fit = eop.fit_powerlaw(dur_sec, xmin='min', xmax = None, discrete = False)
    alpha = dur_fit['power_law_exp']
    alpha_dist = dur_fit['best_fit']
    alpha_dist_TR = dur_fit['T_R_sum']


    #################################
    #    Third   and DCC            #
    #################################

    #third_bin = eop.fit_third_exponent(sizes, dur_bin, discrete= True)
    third = eop.fit_third_exponent(sizes, dur_sec, discrete= False, method = 'pl')

    #dcc_cn_bin = eop.dcc(tau, alpha_bin, third_bin)
    dcc_cn = eop.dcc(tau, alpha, third)


    #################################
    #    REPERTPOIRE               #
    #################################

    # Estimate avalanche functional repertoire
    repertoire = eop.avl_repertoire(avls)
    # normalize the repertoire by signal length
    rep_size = repertoire.shape[0]/sig_length
    
    # rep_similarity_mat = eop.avl_pattern_similarity(repertoire, norm=True)
    rep_similarity_mat = eop.avl_pattern_dissimilarity(repertoire, norm=True)
    rep_dissimilarity_avg = np.mean(rep_similarity_mat)

    #################################
    #    Branching Ratio            #
    #################################
    # Calculate avalanche branching ratio
    avl_br = eop.avl_branching_ratio(avls)
    # Calculate branching ratio
    br = eop.branching_ratio(events_one_chan, BRANCHING_RATIO_TIME_BIN, fs)


    #################################
    #   Susceptibility              #
    #################################

    # Calculate Fano factor
    fano = eop.fano_factor(events_one_chan)

    # Calculate susceptibility
    chi_test, _ = eop.susceptibility(events_by_chan,test = True)
    chi_notest, _ = eop.susceptibility(events_by_chan,test = False)

    ## Save output
    for name in out.keys():
        out[name].append(locals()[name])

    # df_out = pd.DataFrame(out)

    return out, avls