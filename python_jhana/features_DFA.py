#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy import signal
import pandas as pd
from mne.time_frequency import psd_array_multitaper, psd_array_welch
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
import glob
import re
from joblib import Parallel, delayed



# call:  python features_DFA.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -lfrequ 8 -hfrequ 14

def get_channel_hurst(ch_data,sfreq):

    scale = nk.expspace(1*sfreq, 20*sfreq, 40, base=2).astype(np.int64)

    analytic_signal = hilbert(ch_data)
    amplitude_envelope = np.abs(analytic_signal)


    hurst_fh, _ = nk.fractal_hurst(amplitude_envelope, scale=scale, show=False)
    hurst_dfa, _ = nk.fractal_dfa(amplitude_envelope, scale=scale, show=False)

    # try:
    #     hurst_fh, _ = nk.fractal_hurst(amplitude_envelope, scale=scale, show=False)
    # except:
    #     hurst_fh = float('nan')

    # try:
    #     hurst_dfa, _ = nk.fractal_dfa(amplitude_envelope, scale=scale, show=False)
    # except:
    #     hurst_dfa = float('nan')

    return  hurst_fh, hurst_dfa



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Detrended Fluctuation Analysis')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    # parser.add_argument('-part_info', type=str, action='store',
    #                     help='path to txt with information about participants')
    parser.add_argument('-lfrequ', action='store',
                        help='frequency to highpass filter ')
    parser.add_argument('-hfrequ', action='store',
                        help='frequency to lowpass filter')


    # read out arguments
    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir
    lfreq = float(args.lfrequ)
    hfreq = float(args.hfrequ)

    # output
    HURST_FH = []
    HURST_DFA = []
    controls = []
    subs = []
    days = []
    conditions = []
    num_epochs_list = []
    lengths = []
    bad_channels_list = []

    # make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load patient info and conditions
    # info = pd.read_csv(args.part_info,sep = ',', index_col=None)
    # P_IDS = info['ID']
    # Cond = info['Cond']
    # Drug = info['Drug']

    #loop over all conditions and particiants
    paths = glob.glob(f"{in_dir}/*.fif")
    paths.sort()
    for path in paths: 
        fname = os.path.basename(path)

        fname_parts = fname.replace('.fif', '').split('_')

        if len(fname_parts) != 8 or not fname.endswith('_raw.fif'):
            print(f"Skipping invalid file name: {fname}")
            continue

        control      = fname_parts[0]
        sub          = fname_parts[1]
        day          = fname_parts[2]
        condition    = fname_parts[3]
        num_epochs   = fname_parts[4]
        length       = fname_parts[5]
        bad_channels = fname_parts[6]

        print(f"Analyzing Avalanche for subject {sub} - day {day} - condition {condition}")

        # Append metadata
        controls.append(control)
        subs.append(sub)
        days.append(day)
        conditions.append(condition)
        num_epochs_list.append(num_epochs)
        lengths.append(length)
        bad_channels_list.append(bad_channels)



        # for i, p_id in enumerate(P_IDS):
        # print(f"Analyzing DFA of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        # load 1 s epochs
        raw = mne.io.read_raw_fif(path, preload=True)
        raw.pick_types(eeg=True)
        raw.drop_channels(raw.info['bads'])
        data = raw.get_data()        
    
        fs = 256
        sig_length = min(data.shape[1]/fs , 900)
        nr_channels =  data.shape[0]

        # cut data and only use first 200s or less
        cut = int(sig_length*fs)
        data = data[:,:cut]

        data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=lfreq, h_freq=hfreq,verbose=False)

        input = []

        for ch in range(nr_channels):
            input.append((data_filt[ch,:],fs))
            #get_channel_hurst(data_filt[ch,:],fs)

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(get_channel_hurst,input)
        pool.close()
    


        HURST_FH.append(np.mean(pd.DataFrame(results)[0]))
        HURST_DFA.append(np.mean(pd.DataFrame(results)[1]))
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'control': controls,
            'sub': subs,
            'day': days,
            'condition': conditions,
            'num_epochs': num_epochs_list,
            'length': lengths,
            'bad_channels': bad_channels_list,
            'HURST_FH': HURST_FH,
            'HURST_DFA': HURST_DFA
        })
        output_df.to_csv(f'{out_dir}/DFA_{lfreq}_{hfreq}Hz.csv', index=False, sep=',')