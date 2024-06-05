#!/usr/bin/env python
from mne.time_frequency import psd_multitaper, psd_welch
from scipy.signal import hilbert
from scipy.signal import welch
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
from fooof import FOOOF
import edgeofpy as eop
import METHODS_chaos
import pandas as pd
import numpy as np
import argparse
import pickle
import mne.io
import mne
import os




# call:  python features_power.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Complexity using different methods')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')

    # read out arguments
    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir

    # output
    out = {'delta_psd_mean':[],
           'theta_psd_mean':[],
           'alpha_psd_mean':[],
           'beta_psd_mean':[],
           'gamma_psd_mean':[],
           }

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
        print(f"Analyzing Power of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        input_fname = f"{in_dir}/fulldata_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"
        data = loadmat(input_fname)
        data = data['dataOnlyGoodDataPoints']
        fs = 1450
        sig_length = min(data.shape[1]/fs , 300)
        nr_channels =  data.shape[0]

        # cut data and only use first 5 min or less
        cut = np.int(sig_length*fs)
        data = data[:,:cut]

        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=fs,  fmin=0.1,  fmax=4.0, verbose=False,n_fft=5*fs)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=fs,  fmin=4.0,  fmax=8.0, verbose=False,n_fft=5*fs)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=fs,  fmin=8.0, fmax=12.0, verbose=False,n_fft=5*fs)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=fs, fmin=12.0, fmax=30.0, verbose=False,n_fft=5*fs)
        gamma_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=fs, fmin=30.0, fmax=45.0, verbose=False,n_fft=5*fs)

        whole_psd,  fr = mne.time_frequency.psd_array_welch(data, sfreq=fs, fmin=0.1, fmax=45.0, verbose=False,n_fft=5*fs)
        breakpoint()

        delta_psd_mean = np.nanmean(delta_psd, axis=(0,1))
        theta_psd_mean = np.nanmean(theta_psd, axis=(0,1))
        alpha_psd_mean = np.nanmean(alpha_psd, axis=(0,1))
        beta_psd_mean  = np.nanmean(beta_psd, axis=(0,1))
        gamma_psd_mean  = np.nanmean(gamma_psd, axis=(0,1))

        ## Save output
        for name in out.keys():
            out[name].append(locals()[name])

        output_df = pd.DataFrame(out)
        output_2 = pd.DataFrame({'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]})
        tosave = pd.concat((output_2,output_df),axis = 1)
        tosave.to_csv(f'{out_dir}/Powers.txt', index=False, sep=',')

    # save full
    output_df = pd.DataFrame(out)
    output_2 = pd.DataFrame({'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]})
    tosave = pd.concat((output_2,output_df),axis = 1)
    tosave.to_csv(f'{out_dir}/Powers.txt', index=False, sep=',')
