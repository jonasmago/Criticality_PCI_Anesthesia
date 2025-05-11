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
from mne.time_frequency import psd_multitaper, psd_welch
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Edge of Synchrony using different methods')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('-minfreq', type=float, action='store',
                        help='Lower edge of filter frequentcy')
    parser.add_argument('-maxfreq', type=float, action='store',
                        help='Upper edge of filter frequentcy')

    # read out arguments
    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir

    # output
    PCF_mean = []
    OR_mean = []

    # make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load patient info and conditions
    info = pd.read_csv(args.part_info,sep = ',', index_col=None)
    P_IDS = info['ID']
    Cond = info['Cond']
    Drug = info['Drug']

    #loop over all conditions and particiants
    for i, p_id in enumerate(P_IDS):
        print(f"Analyzing Synchrony of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        input_fname = f"{in_dir}/epochs_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"

        data = loadmat(input_fname)
        epochs = data['trails']
        fs = epochs[0].shape[1]/10

        # create info and make epochs a MNE EPOCHS
        info = mne.create_info(ch_names=60, sfreq=fs, ch_types='eeg')
        epochs_mne = mne.EpochsArray(epochs, info)

        # prepare data
        epochs_res = epochs_mne.resample(250)
        epochs_filt = epochs_res.filter(args.minfreq, args.maxfreq, verbose = False)

        # if data is too long only use the first 3 min of data
        nr_trials = min([len(epochs_filt),30]);
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

        PCF_mean.append(np.median(pd.DataFrame(results)[0]))
        OR_mean.append(np.median(pd.DataFrame(results)[1]))

        # save dataframe
        output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1], 'PCF_mean':PCF_mean,'OR_mean':OR_mean}
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{out_dir}/EOS_{args.minfreq}_{args.maxfreq}.txt', index=False, sep=',')

    # save dataframe
    output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1], 'PCF_mean':PCF_mean,'OR_mean':OR_mean}
    output_df = pd.DataFrame(output_df)
    output_df.to_csv(f'{out_dir}/EOS_{args.minfreq}_{args.maxfreq}.txt', index=False, sep=',')
