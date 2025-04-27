# ====== Built-in ======
import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import glob
import time
from datetime import datetime
import pytz

# ====== Scientific Python ======
import numpy as np
from numpy import (absolute, angle, arctan2,
                   diff, exp, imag, mean, real, square,
                   var)
import pandas as pd
import scipy
from scipy import signal, stats
from scipy.fft import fft, ifft
from scipy.fftpack import rfft, irfft
from scipy.io import loadmat, savemat
from scipy.signal import butter, firls, hilbert, lfilter, welch
from scipy.interpolate import interp1d

# ====== External Libraries ======
import mne
import mne.io
import neurokit2 as nk
from fooof import FOOOF
from antropy import lziv_complexity
import antropy
from sklearn import *
import edgeofpy as eop

# from utils.saving import update_results_table, path_to_names
# from features_EOC import features_EOC
# from features_EOS import features_EOS
# from features_Pred import features_Pred
# from features_slope import features_slope
# from features_DFA import features_DFA
# from features_AVC import features_AVC
# from features_avc_std_dist import features_avc_std_dist

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

np.int = int

# ========== CONFIG ========== #
MAX_TRIALS = 200
MAX_S = 2000

# Flags to enable/disable analyses
RUN_EOC = False
RUN_EOS = False
RUN_PRED = False
RUN_SLOPE = True
RUN_DFA = False
RUN_AVC = False
RUN_STD_DIST = False
RUN_ANTROPY = False
RUN_BANDPOWER = False
RUN_SLOPE_PSD = True


# RUN_EOC = True
# RUN_EOS = True
# RUN_PRED = True
# RUN_SLOPE = True
# RUN_DFA = True
# RUN_AVC = True
# RUN_STD_DIST = True
# RUN_ANTROPY = True
# RUN_BANDPOWER = True

###################################################
###################################################

# call:  python features_avc_std_dist.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt
def features_avc_std_dist (raw, max_s=300, fs=256, lfreq=0.5, hfreq=40):

    FIL_FREQ = (lfreq, hfreq) # bandpass frequencies
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


###################################################
def features_AVC (raw, bin_threshold, max_iei, fs=256, max_s=200, lfreq=0.5, hfreq=40):


    FIL_FREQ = (lfreq, hfreq) # bandpass frequencies
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
###################################################

def get_channel_hurst(ch_data,sfreq):
    scale = nk.expspace(1*sfreq, 20*sfreq, 40, base=2).astype(np.int64)
    scale = nk.expspace(1*sfreq, 3*sfreq, 5, base=2).astype(np.int64)

    analytic_signal = hilbert(ch_data)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = (amplitude_envelope - np.mean(amplitude_envelope)) / np.std(amplitude_envelope) # I added this to make DFA same as FH
    
    try:
        hurst_fh, _ =   nk.fractal_hurst(amplitude_envelope, scale=scale, show=False)
    except:
        hurst_fh = np.nan

    try:
        hurst_dfa, _ = nk.fractal_dfa(amplitude_envelope, scale=scale, show=False)
    except:
        hurst_dfa = np.nan

    return hurst_fh, hurst_dfa

def features_DFA(raw, lfreq, hfreq, fs=256, max_s=2000, bad_indices=None):
        
        data = raw.get_data()
        nr_channels =  data.shape[0]

        # cut data and only use first 200s or less
        sig_length = min(data.shape[1]/fs , max_s)
        cut = int(sig_length*fs)
        data = data[:,:cut]

        data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=lfreq, h_freq=hfreq,verbose=False)

        input = []
        results = []
        
        for ch in range(nr_channels):
            input.append((data_filt[ch,:],fs))
            
            # hurst_fh, hurst_dfa = get_channel_hurst(data_filt[ch, :], fs)
            # results.append((hurst_fh, hurst_dfa))


        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_cpus)
        results = pool.starmap(get_channel_hurst,input)
        pool.close()
        pool.join()

        

        print ('## one round done ##')
        results = np.array(results)
        results_interpolated = results.copy()
        results[bad_indices,:] = np.nan

        HURST_FH = np.nanmean(results[:,0])
        HURST_DFA = np.nanmean((results)[:,1])
        HURST_FH_interpolated = np.mean(results_interpolated[:,0])
        HURST_DFA_interpolated = np.mean((results_interpolated)[:,1])

        
        return HURST_FH, HURST_DFA, results, HURST_FH_interpolated, HURST_DFA_interpolated, results_interpolated

###################################################
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
        K_tmp = chaos_pipeline(ch_filt)
        K_ch.append(K_tmp)
        hfreq.append(lpfrequency)
        if type(K_tmp) != np.nan:
            failed.append(0)
        else:
            failed.append(1)

    # print(f'Done Trial {str(trial)} Fixed {str(lpfrequency)} Hz')

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
            K_ch.append(chaos_pipeline(ch_filt))
            hfreq.append(hfreq_tmp)
    # print('Done Trial {}'.format(str(trial)))

    return K_ch, hfreq, failed


def features_EOC(mne_epochs, k_type='flex', hfrequ=None, max_trials=30, bad_indices=None, good_indices=None):

    # output
    Freq = []
    Nopeak = []
    K_median = []
    K_space = []
    
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

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_cpus)
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
    pool.join()

    return K_median, K_median_interpolated, Freq, Nopeak, results_array, results_array_interpolated


###################################################

def calculate_values(trial, epochs):

    data_tr = epochs[trial].get_data()[0]

    #PLE = Methods_EOS.ple(data_tr, m = 5, tau = 2)
    #PLI = Methods_EOS.pli(data_tr)
    PCF, OR_mean, orph_vector_tr, orpa_vector_tr = pcf(data_tr)
    # print(f'Done Trial {str(trial)}')

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

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_cpus)
        # loop over every time segment

        # prepare input for parallel function
        input = []
        for trial in range(nr_trials):
            input.append((trial,epochs_filt))
       
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_cpus)
        results = pool.starmap(calculate_values,input)
        pool.close()
        pool.join()


        

        PCF_mean = np.median(pd.DataFrame(results)[0])
        OR_mean = np.median(pd.DataFrame(results)[1])

        return PCF_mean, OR_mean, results


###################################################
def get_pred(trial, epochs):
    Lyaps = []
    Dims = []
    Ent = []
    LZC = []
    KDF = []

    nr_channels =  epochs.shape[1]
    trial_data = epochs[trial]

    for ch in range(nr_channels):
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

    # print('Done Trial {}'.format(str(trial)))

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

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_cpus)

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
    pool.join()

    
    return Lyaps_max, Dims_mean, Ent_mean, LZC_mean, KDF_mean, Lyaps_max_int, Dims_mean_int, Ent_mean_int, LZC_mean_int, KDF_mean_int, results, results_int 

###################################################

def features_slope (mne_epochs, lfreq, hfreq, fs=256, max_trials=30, bad_indices=None): 
    
    epochs = mne_epochs.get_data()
    nr_trials = min([len(epochs), max_trials])
    nr_channels =  epochs.shape[1]

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




def features_slope_bandpower(mne_epochs, lfreq, hfreq, fs=256, max_trials=30, bad_indices=None): 
    epochs = mne_epochs.get_data()
    nr_trials = min([len(epochs), max_trials])
    nr_channels = epochs.shape[1]

    # Set the frequency range to fit the model
    freq_range = [lfreq, hfreq]
    data_con = np.concatenate(epochs, axis=1)
    
    # get PSD of all channels
    freqs, psds = signal.welch(data_con, fs, nperseg=5*1024)
    psds_interpolated = psds.copy()

    # === Global Slope Estimation ===
    psds_mean_interpolated = np.mean(psds_interpolated, axis=0)
    fm = FOOOF()
    fm.fit(freqs, psds_mean_interpolated, freq_range)
    slope_id_interpolated = -fm.aperiodic_params_[1]
    
    psds[bad_indices, :] = np.nan
    psds_mean = np.nanmean(psds, axis=0)
    fm_nointerp = FOOOF()
    fm_nointerp.fit(freqs, psds_mean, freq_range)
    slope_id = -fm_nointerp.aperiodic_params_[1]

    # === Channel-wise Slope Estimation and Correction ===
    Slope_space_id = []
    psds_interpolated_corrected = np.zeros_like(psds_interpolated)

    for ch in range(psds_interpolated.shape[0]):
        fm_ch = FOOOF()
        fm_ch.fit(freqs, psds_interpolated[ch, :], freq_range)
        slope_ch = -fm_ch.aperiodic_params_[1]
        Slope_space_id.append(slope_ch)

        # Subtract aperiodic background per channel
        background_ch = fm_ch._ap_fit
        
        interp_func = interp1d(fm_ch.freqs, background_ch, bounds_error=False, fill_value="extrapolate")
        background_ch_full = interp_func(freqs)

        psds_interpolated_corrected[ch, :] = psds_interpolated[ch, :] - background_ch_full
        psds_interpolated_corrected[ch, :] = np.clip(psds_interpolated_corrected[ch, :], a_min=0, a_max=None)

    Slope_space_id = np.array(Slope_space_id)
    Slope_space_id_interpolated = Slope_space_id.copy()
    Slope_space_id[bad_indices] = np.nan

    # === Bandpower Computation ===
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    def compute_bandpower(psd, freqs, bands):
        bandpower_per_band = {}
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            bandpower_per_band[band_name] = np.mean(psd[:, freq_mask], axis=1)  # mean across frequencies
        return bandpower_per_band

    bandpower_raw_per_channel = compute_bandpower(psds_interpolated, freqs, bands)
    bandpower_corrected_per_channel = compute_bandpower(psds_interpolated_corrected, freqs, bands)

    bandpower_raw_avg = {k: np.nanmean(v) for k, v in bandpower_raw_per_channel.items()}
    bandpower_corrected_avg = {k: np.nanmean(v) for k, v in bandpower_corrected_per_channel.items()}

    # === Final Outputs ===
    return (slope_id, slope_id_interpolated, 
            Slope_space_id, Slope_space_id_interpolated, 
            np.array(psds_mean), np.array(psds_mean_interpolated), 
            np.array(psds), np.array(psds_interpolated), np.array(freqs),
            bandpower_raw_per_channel, bandpower_corrected_per_channel,
            bandpower_raw_avg, bandpower_corrected_avg)





###################################################
## Methods Chaos
def chaos_pipeline(data, sigma=0.5, denoise=False, downsample='minmax'):
    """Simplified pipeline for the modified 0-1 chaos test emulating the
    implementation from Toker et al. (2022, PNAS). This test assumes
    that the signal is stationarity and deterministic.

    Parameters
    ----------
    data : 1d array
        The (filtered) signal.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    denoise : bool
        If True, denoising will be applied according to the method by Schreiber
        (2000).
    downsample : str or bool
        If 'minmax', signal will be downsampled by conserving only local minima
        and maxima.

    Returns
    -------
    K: float
        Median K-statistic.

    """
    if denoise:
        # Denoise data using Schreiber denoising algorithm
        data = schreiber_denoise(data)

    if downsample == 'minmax':
        # Downsample data by preserving local minima
        data = _minmaxsig(data)

    # Check if signal is long enough, else return NaN
    if len(data) < 20:
        return np.nan

    # Normalize standard deviation of signal
    x = data * (0.5 / np.std(data)) # matlab equivalent  np.std(data,ddof=1)

    # Mdified 0-1 chaos test
    K = z1_chaos_test(x, sigma=sigma)

    return K


def z1_chaos_test(x, sigma=0.5, rand_seed=0):
    """Modified 0-1 chaos test. For long time series, the resulting K-statistic
    converges to 0 for regular/periodic signals and to 1 for chaotic signals.
    For finite signals, the K-statistic estimates the degree of chaos.

    Parameters
    ----------
    x : 1d array
        The time series.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    rand_seed : int, optional
        Seed for random number generator. The default is 0.

    Returns
    -------
    median_K : float
        Indicator of chaoticity. 0 is regular/stable, 1 is chaotic and values
        in between estimate the degree of chaoticity.

    References
    ----------
    Gottwald & Melbourne (2004) P Roy Soc A - Math Phy 460(2042), 603-11.
    Gottwald & Melbourne (2009) SIAM J Applied Dyn Sys 8(1), 129-45.
    Toker et al. (2022) PNAS 119(7), e2024455119.
    """
    np.random.seed(rand_seed)
    N = len(x)
    j = np.arange(1,N+1)
    t = np.arange(1,int(round(N / 10))+1)
    M = np.zeros(int(round(N / 10)))
    # Choose a coefficient c within the interval pi/5 to 3pi/5 to avoid
    # resonances. Do this 1000 times.
    c = np.pi / 5 + np.random.random_sample(1000) * 3 * np.pi / 5
    k_corr = np.zeros(1000)

    for its in range(1000):
        # Create a 2-d system driven by the data
        #p = cumsum(x * cos(a * c[i]))
        #q = cumsum(x * sin(a * c[i]))
        p=np.cumsum(x * np.cos(j*c[its]))
        q=np.cumsum(x * np.sin(j*c[its]))

        for n in t:
            # Calculate the (time-averaged) mean-square displacement,
            # subtracting a correction term (Gottwald & Melbourne, 2009)
            # and adding a noise term scaled by sigma (Dawes & Freeland, 2008)

            #M[n-1]=(np.mean((p[n+1:N] - p[1:N-n])**2 + (q[n+1:N]-q[1:N-n])**2)
            #      - np.mean(x)**2 * (1-np.cos(n*c[its])) / (1-np.cos(c[its]))
            #      + sigma * (np.random.random()-.5))

            M[n-1]=(np.mean((p[n:N] - p[:N-n])**2 + (q[n:N]-q[:N-n])**2)
                  - np.mean(x)**2 * (1-np.cos(n*c[its])) / (1-np.cos(c[its]))
                  + sigma * (np.random.random()-.5))

        k_corr[its], _ = scipy.stats.pearsonr(t, M)
        median_k = np.median(k_corr)

    return median_k

def _minmaxsig(x):
    maxs = scipy.signal.argrelextrema(x, np.greater)[0]
    mins = scipy.signal.argrelextrema(x, np.less)[0]
    minmax = np.concatenate((mins, maxs))
    minmax.sort(kind='mergesort')
    return x[minmax]

###################################################
## Methods EOS

def pcf(data):
    """Estimate the pair correlation function (PCF) in a network of
    oscillators, equivalent to the susceptibility in statistical physics.
    The PCF shows a sharp peak at a critical value of the coupling between
    oscillators, signaling the emergence of long-range correlations between
    oscillators.

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.

    Returns
    -------
    pcf : float
        The pair correlation function, a scalar value >= 0.
    orpa: float
        Absolute value of the order parameter (degree of synchronicity)
        averaged over time, being equal to 0 when the oscillators’ phases are
        uniformly distributed in [0,2π ) and 1 when they all have the same
        phase.
    orph_vector: 1d array (length = N_timepoints)
        Order parameter phase for every moment in time.
    orpa_vector: 1d array (length = N_timepoints)
        Absolute value of the order parameter (degree of synchronicity) for
        every moment in time.

    References
    ----------
    Yoon et al. (2015) Phys Rev E 91(3), 032814.
    """

    N_ch = min(data.shape)  # Nr of channels
    N_time = max(data.shape)  # Nr of channels

    # inifialize empty array
    inst_phase = np.zeros(data.shape)
    z_vector = []

    # calculate Phase of
    for i in range(N_ch):
        inst_phase[i,:] = np.angle(scipy.signal.hilbert(data[i,:]))

    for t in range(N_time):
        # get global synchronization order parameter z over time
        z_vector.append(np.mean(exp(1j * inst_phase[:,t])))

    z_vector = np.array(z_vector)

    #  r =|z| degree of synchronicity
    orpa_vector = abs(z_vector)
    # get order phases
    orph_vector = arctan2(imag(z_vector), real(z_vector))

    # get PCF = variance of real part of order parameter
    # var(real(x)) == (mean(square(real(x))) - square(mean(real(x))))
    pcf = N_ch * var(real(z_vector))
    # time-averaged Order Parameter
    orpa = mean(orpa_vector);

    return pcf, orpa, orph_vector, orpa_vector


###################################################
## Saving

def path_to_names(path):
    fname = os.path.basename(path)
    fname_parts = fname.replace('.fif', '').split('_')
    if len(fname_parts) != 12 or not fname.endswith('.fif'):
        print(f"Skipping invalid file name: {fname}")
        return None
    control         = fname_parts[0]
    sub             = fname_parts[1]
    day             = fname_parts[2]
    condition       = fname_parts[3]
    num_bad_channels= fname_parts[4]
    n_elem_raw      = fname_parts[5]
    length_raw      = fname_parts[6]
    n_epochs_10     = fname_parts[7]
    length_10       = fname_parts[8]
    n_epochs_3      = fname_parts[9]
    length_3        = fname_parts[10]

    return control, sub, day, condition, num_bad_channels, n_elem_raw, length_raw, n_epochs_10, length_10, n_epochs_3, length_3



def update_results_table(path, row_dict, results_table_path, results_dict_dir, dict_outputs=None):
    # Load or create table
    if os.path.exists(results_table_path):
        df = pd.read_csv(results_table_path)
    else:
        df = pd.DataFrame()

    # Get identifier for the current file
    path_id = os.path.basename(path)
    row_dict['path'] = path_id  # Ensure path is included

    # ✅ Ensure 'path' column exists in df
    if 'path' not in df.columns:
        df['path'] = pd.NA  # or np.nan

    if path_id in df['path'].values:
        for key, value in row_dict.items():
            if key not in df.columns:
                df[key] = 'NA'
            if value != 'NA':
                df.loc[df['path'] == path_id, key] = value
    else:
        for key in row_dict:
            if key not in df.columns:
                df[key] = 'NA'
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    df.to_csv(results_table_path, index=False)

    # Save detailed dictionary outputs
    if dict_outputs:
        pkl_path = os.path.join(results_dict_dir, f"{path_id}.pkl")

        # Load existing data if it exists
        if os.path.exists(pkl_path):
            try: 
                with open(pkl_path, 'rb') as f:
                    existing_data = pickle.load(f)
            except: 
                existing_data = {}
        else:
            existing_data = {}

        # Update with new values
        existing_data.update(dict_outputs)

        # Save the updated version
        with open(pkl_path, 'wb') as f:
            pickle.dump(existing_data, f)

    print('saving is done')







def get_antropy_measures(trial, epochs):
    ant_lziv = []
    ant_perm_entropy = []
    ant_spectral_entropy = []
    ant_sample_entropy = []
    ant_hjorth_mobility = []
    ant_hjorth_complexity = []

    nr_channels = epochs.shape[1]
    trial_data = epochs[trial]

    for ch in range(nr_channels):
        time_series = trial_data[ch]

        threshold = np.median(time_series)
        binary_sequence = (time_series > threshold).astype(int)
        binary_string = ''.join(binary_sequence.astype(str))

        ant_lziv.append(lziv_complexity(binary_string, normalize=True))
        ant_perm_entropy.append(antropy.perm_entropy(time_series, normalize=True))
        ant_spectral_entropy.append(antropy.spectral_entropy(time_series, sf=256, method='welch', normalize=True))
        ant_sample_entropy.append(antropy.sample_entropy(time_series))

        mobility, complexity = antropy.hjorth_params(time_series)
        ant_hjorth_mobility.append(mobility)
        ant_hjorth_complexity.append(complexity)

    return (ant_lziv, ant_perm_entropy, ant_spectral_entropy, ant_sample_entropy, ant_hjorth_mobility, ant_hjorth_complexity)


def features_Antropy(mne_epochs, lfreq=0.5, hfreq=45, fs=256, max_trials=30, bad_indices=None):
    epochs_res = mne_epochs
    epochs_filt = epochs_res.filter(lfreq, hfreq, verbose=False)

    nr_trials = min(len(epochs_filt), max_trials)
    epochs = epochs_filt.get_data()
    nr_channels = epochs.shape[1]

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_cpus)

    input = []
    for trial in range(nr_trials):
        input.append((trial, epochs))

    results = pool.starmap(get_antropy_measures, input)

    results = np.array(results)
    if bad_indices is not None:
        results[:, :, bad_indices] = np.nan

    results_int = np.array(results)

    # Average across trials and channels
    ant_lziv = np.nanmean(results[:, 0, :])
    ant_perm_entropy = np.nanmean(results[:, 1, :])
    ant_spectral_entropy = np.nanmean(results[:, 2, :])
    ant_sample_entropy = np.nanmean(results[:, 3, :])
    ant_hjorth_mobility = np.nanmean(results[:, 4, :])
    ant_hjorth_complexity = np.nanmean(results[:, 5, :])

    ant_lziv_int = np.nanmean(results_int[:, 0, :])
    ant_perm_entropy_int = np.nanmean(results_int[:, 1, :])
    ant_spectral_entropy_int = np.nanmean(results_int[:, 2, :])
    ant_sample_entropy_int = np.nanmean(results_int[:, 3, :])
    ant_hjorth_mobility_int = np.nanmean(results_int[:, 4, :])
    ant_hjorth_complexity_int = np.nanmean(results_int[:, 5, :])

    pool.close()
    pool.join()

    return (ant_lziv, ant_perm_entropy, ant_spectral_entropy, ant_sample_entropy,
            ant_hjorth_mobility, ant_hjorth_complexity,
            ant_lziv_int, ant_perm_entropy_int, ant_spectral_entropy_int,
            ant_sample_entropy_int, ant_hjorth_mobility_int, ant_hjorth_complexity_int,
            results, results_int)





def get_bandpower_measures(trial, epochs, fs=256):
    delta = []
    theta = []
    alpha = []
    beta = []
    gamma = []

    nr_channels = epochs.shape[1]
    trial_data = epochs[trial]

    freqs, psd = signal.welch(trial_data, fs=fs, nperseg=fs*2, axis=-1)

    psd = psd * 1e12

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    for ch in range(nr_channels):
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            band_power = np.mean(psd[ch, freq_mask])
            if band_name == 'delta':
                delta.append(band_power)
            elif band_name == 'theta':
                theta.append(band_power)
            elif band_name == 'alpha':
                alpha.append(band_power)
            elif band_name == 'beta':
                beta.append(band_power)
            elif band_name == 'gamma':
                gamma.append(band_power)

    return delta, theta, alpha, beta, gamma


def features_Bandpower(mne_epochs, fs=256, max_trials=30, bad_indices=None):
    epochs = mne_epochs.get_data()
    nr_trials = min(len(epochs), max_trials)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_cpus)

    input = []
    for trial in range(nr_trials):
        input.append((trial, epochs))

    results = pool.starmap(get_bandpower_measures, input)

    results = np.array(results)
    if bad_indices is not None:
        results[:, :, bad_indices] = np.nan

    results_int = np.array(results)

    delta = np.nanmean(results[:, 0, :])
    theta = np.nanmean(results[:, 1, :])
    alpha = np.nanmean(results[:, 2, :])
    beta = np.nanmean(results[:, 3, :])
    gamma = np.nanmean(results[:, 4, :])

    delta_int = np.nanmean(results_int[:, 0, :])
    theta_int = np.nanmean(results_int[:, 1, :])
    alpha_int = np.nanmean(results_int[:, 2, :])
    beta_int = np.nanmean(results_int[:, 3, :])
    gamma_int = np.nanmean(results_int[:, 4, :])

    pool.close()
    pool.join()

    return (delta, theta, alpha, beta, gamma,
            delta_int, theta_int, alpha_int, beta_int, gamma_int,
            results, results_int)




###################################################
###################################################


if __name__ == "__main__":
    print("script started")
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {n_cpus} CPUs from SLURM allocation")


    parser = argparse.ArgumentParser(description='Calculate Edge of Synchrony using different methods')
    parser.add_argument('-device', type=str, action='store',
                        help='decide if this script runs local (l) or ona cluster (c)')
    parser.add_argument('-name', type=str, action='store',
                        help='name to specify output')
    args = parser.parse_args()
    device = args.device
    name = args.name

    if device == 'l':
        results_table_path = f'output/results_l_{name}/summary.csv'
        results_dict_dir = f'output/results_l_{name}/details/'
        os.makedirs(results_dict_dir, exist_ok=True)
        paths = glob.glob('/Users/jonasmago/PhD_code_data/github/eeg_jhana/notebooks/hand_cleaning/ALL/03s/*.fif')
        paths.sort()

    if device == 'c':
        results_table_path = f'output/results_c_{name}/summary.csv'
        results_dict_dir = f'output/results_c_{name}/details/'
        os.makedirs(results_dict_dir, exist_ok=True)
        paths = glob.glob('/home/jmago/projects/def-michael9/jmago/jhana_eeg/data/10s/*.fif')
        paths.sort()
        # print (paths)

    start = 0
    for path_i_relative, path in enumerate(paths[start:]):

        path_i = path_i_relative+start
        print(f"\n>>> Processing {os.path.basename(path)}")
        t_start = time.time()
        
        # Store metadata
        row_data, dict_data = {}, {}
        control, sub, day, condition, num_bad_channels, n_elem_raw, length_raw, n_epochs_10, length_10, n_epochs_3, length_3 = path_to_names(path)
        row_data.update({
            'path_i': path_i,
            'control': control, 'sub': sub, 'day': day, 'condition': condition,
            'num_bad_channels': num_bad_channels, 'n_elem_raw': n_elem_raw,
            'length_raw': length_raw, 'n_epochs_10': n_epochs_10, 'length_10': length_10,
            'n_epochs_3': n_epochs_3, 'length_3': length_3
        })

        # Load Epoch + Raw
        mne_epochs_raw = mne.read_epochs(path, preload=True)
        if len(mne_epochs_raw) < 1:
            print ('### skip, no valid epoch ###')
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)
            continue

        mne_epochs_32 = mne_epochs_raw.copy()
        mne_epochs_raw.pick('eeg')

        if len(mne_epochs_32.info['ch_names']) <  5:
            print ('### skip, less than 5 good channels ###')
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)
            continue

        mne_epochs_32.interpolate_bads(reset_bads=True).pick('eeg')

        bad_chans = mne_epochs_raw.info['bads']
        all_chans = mne_epochs_32.ch_names
        good_chans = [ch for ch in all_chans if ch not in bad_chans]
        channel_indices = {name: idx for idx, name in enumerate(all_chans)}
        bad_indices = [channel_indices[ch] for ch in bad_chans]
        good_indices = [channel_indices[ch] for ch in good_chans]

        path_raw = path.replace('_epo.fif', '_raw.fif').replace('10s', 'raw').replace('03s', 'raw')
        raw_raw = mne.io.read_raw_fif(path_raw, preload=True)
        raw_32 = raw_raw.copy()
        raw_32.interpolate_bads(reset_bads=True).pick('eeg')
        raw_raw.pick('eeg')





        fbands = [[1, 45], [1, 4], [4, 8], [8, 13], [13, 30], [30, 45]]

        # ========== EOC ========== #
        if RUN_EOC:
            print (">>>>> processing EOC <<<<<")
            try:
                K_median, K_median_interpolated, Freq, Nopeak, eoc_results, eoc_results_interpolated = features_EOC(
                    mne_epochs_32, k_type='flex', hfrequ=None, max_trials=MAX_TRIALS,
                    bad_indices=bad_indices, good_indices=good_indices)
                row_data.update({
                    'K_median': K_median, 'K_median_interpolated': K_median_interpolated,
                    'Freq': Freq, 'Nopeak': Nopeak
                })
                dict_data.update({
                    'eoc_results': eoc_results,
                    'eoc_results_interpolated': eoc_results_interpolated
                })
                
                K_median_f4, K_median_interpolated_f4, Freq_f4, Nopeak_f4, eoc_results_f4, eoc_results_interpolated_f4 = features_EOC(
                    mne_epochs_32, k_type='fixed', hfrequ=4, max_trials=MAX_TRIALS,
                    bad_indices=bad_indices, good_indices=good_indices)
                
                row_data.update({
                    'K_median_f4': K_median_f4, 'K_median_interpolated_f4': K_median_interpolated_f4,
                    'Freq_f4': Freq_f4, 'Nopeak_f4': Nopeak_f4
                })
                dict_data.update({
                    'eoc_results_f4': eoc_results_f4,
                    'eoc_results_interpolated_f4': eoc_results_interpolated_f4
                })

            except Exception as e:
                print(f"[EOC] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)

        # ========== EOS ========== #
        if RUN_EOS:
            print (">>>>> processing EOS <<<<<")
            try:
                PCF_mean, OR_mean, eos_results = features_EOS(mne_epochs_32, minfreq=1, maxfreq=10, fs=256, max_trials=MAX_TRIALS)
                row_data.update({'PCF_mean': PCF_mean, 'OR_mean': OR_mean})
                dict_data['eos_results'] = eos_results
            except Exception as e:
                print(f"[EOS] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)

        # ========== PRED ========== #
        if RUN_PRED:
            print (">>>>> processing PRED <<<<<")
            try:
                vals = features_Pred(mne_epochs_32, lfreq=0.5, hfreq=40, fs=256, max_trials=MAX_TRIALS, bad_indices=bad_indices)
                names = ['Lyaps_max', 'Dims_mean', 'Ent_mean', 'LZC_mean', 'KDF_mean', 'Lyaps_max_interpolated', 'Dims_mean_interpolated', 'Ent_mean_interpolated', 'LZC_mean_interpolated', 'KDF_mean_interpolated']
                row_data.update({name: val for name, val in zip(names, vals[:10])})
                dict_data['pred_results'] = vals[10]
                dict_data['pred_results_interpolated'] = vals[11]
            except Exception as e:
                print(f"[PRED] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)

        # ========== SLOPE ========== #
        if RUN_SLOPE:
            print (">>>>> processing Slope <<<<<")
            try:
                vals = features_slope(mne_epochs_32, lfreq=0.5, hfreq=40, fs=256, max_trials=MAX_TRIALS, bad_indices=bad_indices)
                row_data.update({
                    'slope_id': vals[0], 'slope_id_interpolated': vals[1]
                })
                dict_data['slope'] = {'Slope_space_id': vals[2], 
                                    'Slope_space_id_interpoalted': vals[3],
                                    'psds_mean': vals[4],
                                    'psds_mean_interpolated': vals[5],
                                    'psds': vals[6],
                                    'psds_interpoalted': vals[7],
                                    'freqs': vals[8],
                                    }
            except Exception as e:
                print(f"[SLOPE] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)



        # ========== SLOPE PSD ========== #
        if RUN_SLOPE_PSD:
            print(">>>>> processing Slope + Bandpower <<<<<")
            try:
                vals = features_slope_bandpower(mne_epochs_32, lfreq=0.5, hfreq=40, fs=256, max_trials=MAX_TRIALS, bad_indices=bad_indices)
                
                # Save slopes
                row_data.update({
                    'sp_slope_id': vals[0],
                    'sp_slope_id_interpolated': vals[1]
                })

                # Save bandpowers to row_data: both raw and corrected, both from interpolated PSD
                band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

                sp_bandpower_raw_avg = vals[11]       # dict: band name -> mean raw PSD power
                sp_bandpower_corrected_avg = vals[12] # dict: band name -> mean corrected PSD power

                for band in band_names:
                    row_data[f'sp_{band}_raw'] = sp_bandpower_raw_avg.get(band, np.nan)
                    row_data[f'sp_{band}_corrected'] = sp_bandpower_corrected_avg.get(band, np.nan)

                # Save full arrays into dict_data
                dict_data['slope_psd'] = {
                    'sp_slope_space_id': vals[2],
                    'sp_slope_space_id_interpolated': vals[3],
                    'sp_psds_mean': vals[4],
                    'sp_psds_mean_interpolated': vals[5],
                    'sp_psds': vals[6],
                    'sp_psds_interpolated': vals[7],
                    'sp_freqs': vals[8],
                    'sp_bandpower_raw_per_channel': vals[9],
                    'sp_bandpower_corrected_per_channel': vals[10],
                    'sp_bandpower_raw_avg': vals[11],
                    'sp_bandpower_corrected_avg': vals[12]
                }

            except Exception as e:
                import pdb; pdb.set_trace()
                print(f"[SLOPE_PSD] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)


        # ========== DFA ========== #
        if RUN_DFA:
            print (">>>>> processing DFA <<<<<")
            for fband in fbands:
                try:
                    HURST_FH, HURST_DFA, results, HURST_FH_int, HURST_DFA_int, results_interpolated = features_DFA(
                        raw_32, lfreq=fband[0], hfreq=fband[1], fs=256, max_s=MAX_S, bad_indices=bad_indices)
                    band_name = f"{fband[0]}-{fband[1]}Hz"
                    row_data[f'HURST_FH_{band_name}'] = HURST_FH
                    row_data[f'HURST_DFA_{band_name}'] = HURST_DFA
                    row_data[f'HURST_FH_interpolated_{band_name}'] = HURST_FH_int
                    row_data[f'HURST_DFA_interpolated_{band_name}'] = HURST_DFA_int

                    dict_data[f'results_DFA_{band_name}'] = results
                    dict_data[f'results_DFA_int_{band_name}'] = results_interpolated


                except Exception as e:
                    print(f"[DFA {fband}] Error: {e}")
                    # import traceback; traceback.print_exc()
                    # import pdb; pdb.set_trace()
                update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)

        # ========== AVC ========== #
        if RUN_AVC:
            print (">>>>> processing AVC <<<<<")
            try:
                out, avls = features_AVC(raw_32, bin_threshold=0.0005, max_iei=0.2, fs=256, max_s=MAX_S, lfreq=0.5, hfreq=40)
                for key, value in out.items():
                    if isinstance(value, list) and len(value) > 0:
                        row_data[key] = value[0]
                    else:
                        row_data[key] = 'NA'

            except Exception as e:
                print(f"[AVC] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)

        # ========== STD_DIST ========== #
        if RUN_STD_DIST:
            print (">>>>> processing std dist <<<<<")
            try:
                part_hist_x, part_hist_y, part_hist_x_raw, part_hist_y_raw = features_avc_std_dist(raw_32, max_s=MAX_S, fs=256, lfreq=0.5, hfreq=40)
                dict_data['part_hist_x'] = part_hist_x
                dict_data['part_hist_y'] = part_hist_y
                dict_data['part_hist_x_raw'] = part_hist_x_raw
                dict_data['part_hist_y_raw'] = part_hist_y_raw

            except Exception as e:
                print(f"[STD_DIST] Error: {e}")
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)



        # ========== ANTROPY ========== #
        if RUN_ANTROPY:
            print(">>>>> processing Antropy Features <<<<<")
            try:
                vals = features_Antropy(mne_epochs_32, lfreq=0.5, hfreq=45, fs=256, max_trials=MAX_TRIALS, bad_indices=bad_indices)

                # Define output names
                names = [
                    'ant_lziv', 'ant_perm_entropy', 'ant_spectral_entropy', 'ant_sample_entropy',
                    'ant_hjorth_mobility', 'ant_hjorth_complexity',
                    'ant_lziv_int', 'ant_perm_entropy_int', 'ant_spectral_entropy_int', 'ant_sample_entropy_int',
                    'ant_hjorth_mobility_int', 'ant_hjorth_complexity_int'
                ]

                # First 12 outputs go into row_data
                row_data.update({name: val for name, val in zip(names, vals[:12])})

                # The last two are the detailed results arrays
                dict_data['antropy_results'] = vals[12]
                dict_data['antropy_results_int'] = vals[13]

            except Exception as e:
                print(f"[ANTROPY] Error: {e}")
                import pdb; pdb.set_trace()
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)





        if RUN_BANDPOWER:
            print(">>>>> processing Band Powers <<<<<")
            try:
                vals = features_Bandpower(mne_epochs_32, fs=256, max_trials=MAX_TRIALS, bad_indices=bad_indices)
                names = [
                    'delta', 'theta', 'alpha', 'beta', 'gamma',
                    'delta_int', 'theta_int', 'alpha_int', 'beta_int', 'gamma_int'
                ]
                row_data.update({name: val for name, val in zip(names, vals[:10])})
                dict_data['bandpower_results'] = vals[10]
                dict_data['bandpower_results_int'] = vals[11]
            except Exception as e:
                print(f"[BANDPOWER] Error: {e}")
            update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)



        # Compute elapsed time and current time in EST
        elapsed_time_sec = time.time() - t_start
        current_time_est = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")

        # Add to row_data
        row_data['MAX_TRIALS'] = MAX_TRIALS
        row_data['MAX_S'] = MAX_S
        row_data['elapsed_time_sec'] = round(elapsed_time_sec, 2)
        row_data['timestamp_est'] = current_time_est

        # Save results
        update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)
        print(f"⏱️ Elapsed time for file: {elapsed_time_sec:.2f} seconds (finished at {current_time_est} EST)")
