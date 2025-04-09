#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy import signal
import METHODS_chaos
import pandas as pd
from mne.time_frequency import psd_multitaper, psd_welch
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Slope')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('-lfrequ', action='store',
                        help='frequency of lower end of the aperiodic component estimation ')
    parser.add_argument('-hfrequ', action='store',
                        help='frequency of higher end of the aperiodic component estimation')


    # read out arguments
    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir
    lfreq = float(args.lfrequ)
    hfreq = float(args.hfrequ)

    # output
    Slope = []
    Slope_space = []

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
        print(f"Analyzing Slope of {p_id}")

        #################################
        #          LOAD  DATA          #
        #################################

        input_fname = f"{in_dir}/epochs_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"

        data = loadmat(input_fname)
        epochs = data['trails']
        fs = epochs[0].shape[1]/10

        # if data is too long only use the first 3 min of data
        nr_trials = min([len(epochs),30])
        nr_channels =  epochs.shape[1]

        # search individual lowpass frequency
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [lfreq, hfreq]
        data_con = np.concatenate(epochs,axis = 1)
        # get psd of channels
        freqs, psds = signal.welch(data_con,fs,nperseg=5*1024)

        # Get average Slope
        psds_mean =  np.mean(psds,axis = 0)
        fm.fit(freqs, psds_mean, freq_range)
        slope_id = -fm.aperiodic_params_[1]
        Slope.append(slope_id)

        Slope_space_id = []
        # Get Space-resolved Slope
        for ch in range(len(psds)):
            fm.fit(freqs, psds[ch,:], freq_range)
            slope_id = -fm.aperiodic_params_[1]
            Slope_space_id.append(slope_id)

        Slope_space.append(Slope_space_id)

        # save part
        output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1],'Slope':Slope}
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{out_dir}/Slope.txt', index=False, sep=',')

        # save part
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(Slope_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/Slope_space.txt', index=False, sep=',')
