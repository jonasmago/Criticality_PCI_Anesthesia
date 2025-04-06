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
from scipy import stats
import pandas as pd
import numpy as np
import argparse
import pickle
import mne.io
import mne
import os




# call:  python features_avc_std_dist.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Complexity using different methods')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')

    FIL_FREQ = (0.5, 40) # bandpass frequencies
    THRESH_TYPE = 'both' # Fosque22: 'both'

    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir

    # make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load patient info and conditions
    info = pd.read_csv(args.part_info,sep = ',', index_col=None)
    P_IDS = info['ID']
    Cond = info['Cond']
    Drug = info['Drug']

    hist_x = []
    hist_y = []

    hist_x_raw = []
    hist_y_raw = []

    #loop over all conditions and particiants
    for i, p_id in enumerate(P_IDS):
        print(f"Analyzing Avlanches of {p_id}");

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

        data_filt = mne.filter.filter_data(data, sfreq=fs, l_freq=FIL_FREQ[0], h_freq=FIL_FREQ[1],verbose=False)

        # FIND DEVIATIONS FOR RELATIVE STD (Per recording)
        mean_per_chan = np.mean(data_filt, axis=1, keepdims=True)
        std_per_chan = np.std(data_filt, axis=1, keepdims=True)

        # Z-score all data
        data_z = (data_filt-mean_per_chan)/std_per_chan

        scale = np.linspace(-10,10,100)
        part_hist_y, part_hist_x = np.histogram(data_z, bins = scale)

        hist_x.append(part_hist_x)
        hist_y.append(part_hist_y)

        output_df_x = pd.DataFrame(hist_x)
        output_df_Y = pd.DataFrame(hist_y)
        output_2 = pd.DataFrame({'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]})
        tosave = pd.concat((output_2,output_df_x,output_df_Y),axis = 1)
        tosave.to_csv(f'{out_dir}/AVC_std_varley.txt', index=False, sep=',')

        # do the same without z scores
        scale = np.linspace(-100,100,100)
        part_hist_y, part_hist_x = np.histogram(data_filt, bins = scale)

        hist_x_raw.append(part_hist_x)
        hist_y_raw.append(part_hist_y)

        output_df_x = pd.DataFrame(hist_x_raw)
        output_df_Y = pd.DataFrame(hist_y_raw)
        output_2 = pd.DataFrame({'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]})
        tosave = pd.concat((output_2,output_df_x,output_df_Y),axis = 1)
        tosave.to_csv(f'{out_dir}/AVC_std_varley_raw.txt', index=False, sep=',')
