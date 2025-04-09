#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy import signal
import pandas as pd
from mne.time_frequency import psd_multitaper, psd_welch
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

    analytic_signal = hilbert(ch_data)
    amplitude_envelope = np.abs(analytic_signal)

    try:
        hurst_fh, _ = nk.fractal_hurst(amplitude_envelope, scale=scale, show=False)
    except:
        hurst_fh = float('nan')

    try:
        hurst_dfa, _ = nk.fractal_dfa(amplitude_envelope, scale=scale, show=False)
    except:
        hurst_dfa = float('nan')

    return  hurst_fh, hurst_dfa



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Detrended Fluctuation Analysis')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')
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
        print(f"Analyzing DFA of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        # load 1 s epochs
        input_fname = f"{in_dir}/fulldata_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"
        data = loadmat(input_fname)
        data = data['dataOnlyGoodDataPoints']
        fs = 1450
        sig_length = min(data.shape[1]/fs , 200)
        nr_channels =  data.shape[0]

        # cut data and only use first 200s or less
        cut = np.int(sig_length*fs)
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

        # save part
        output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1],'HURST_FH':HURST_FH,'HURST_DFA':HURST_DFA}
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{out_dir}/DFA_{lfreq}_{hfreq}.txt', index=False, sep=',')
