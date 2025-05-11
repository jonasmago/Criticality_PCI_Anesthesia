#!/usr/bin/env python
from scipy.signal import hilbert
from scipy.signal import welch
from scipy.signal import butter, lfilter,firls
from scipy import signal
from utils import METHODS_chaos
import pandas as pd
from scipy import stats
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



# call:  python features_EOC.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -ktype flex

def fixed_chaos(trial, epochs, lpfrequency):
    K_ch = []
    hfreq = []
    failed = []

    # select trial from epoch
    data_trial = epochs[trial]
    fs = data_trial.shape[1]/10
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
    fs = data_trial.shape[1]/10

    nr_channels =  epochs.shape[1]
    for ch in range(nr_channels):
        # select channel data
        data_ch = data_trial[ch,:]
        # do FOOOF to find lowst frequency peak
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [1, 6]
        # get psd of channels
        freqs, psd_ch = signal.welch(data_ch,fs,nperseg=5*1024)

        fm.fit(freqs, psd_ch, freq_range)

        if fm.peak_params_.shape[0] == 0:
            #no peak found, output nan
            failed.append(1)
            hfreq.append( np.NaN )
            K_ch.append( np.NaN )
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Complexity using different methods')
    parser.add_argument('-data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='directory for results to be saved')
    parser.add_argument('-part_info', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('-ktype', action='store', default = 'flex',choices=['fixed', 'flex','indflex'],
                        help='flex applies fooof to search for a low frequency peak. Fixed applies a lowpass filter at the specified frequency ')
    parser.add_argument('-hfrequ', action='store',
                        help='frequency to lowpass filter. Only specify if type == fixed')


    # read out arguments
    args = parser.parse_args()
    out_dir = args.output_dir
    in_dir = args.data_dir
    k_type = args.ktype

    # output
    Freq = []
    Nopeak = []
    K_median = []
    K_space = []

    if k_type == 'fixed':
        hfrequ = args.hfrequ
        k_type = f"fixed_{str(hfrequ)}"

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
        print(f"Analyzing Chaos of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        input_fname = f"{in_dir}/epochs_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"

        data = loadmat(input_fname)
        epochs = data['trails']
        #epochs = stats.zscore(epochs, axis =1)
        fs = epochs[0].shape[1]/10

        #data_test = loadmat('test_data.mat')['data_channel_filt'][0]
        #METHODS_chaos.chaos.chaos_pipeline(data_test)

        # if data is too long only use the first 3 min of data
        nr_trials = min([len(epochs),30]);
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
            elif args.ktype == 'fixed':
                input.append((trial,epochs, np.float64(hfrequ)))

            #filter_and_chaos(trial,epochs)
            #fixed_chaos(trial,epochs, np.float(hfrequ))

        #get results for chaos test
        if k_type == 'flex':
            results = pool.starmap(filter_and_chaos,input)
        else:
            results = pool.starmap(fixed_chaos,input)

        K_median.append(np.nanmedian(np.array(results)[:,0,:]))
        K_space.append(np.nanmedian(np.array(results)[:,0,:],axis = 0))
        Freq.append(np.nanmean(np.array(results)[:,1,:]))
        Nopeak.append(np.sum(np.array(results)[:,2,:])/(nr_trials*nr_channels))

        pool.close()

        # save all lowpass frequencies
        #lp_freqs =np.array(results)[:,1,:]
        #savemat(f"{out_dir}/lpfs/lpf_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat", {'lp_freqs': lp_freqs})

        # save part
        output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1],'K_median':K_median,'Freq':Freq,'Nopeak':Nopeak}
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{out_dir}/01Chaos_{k_type}.txt', index=False, sep=',')

        # save part
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(K_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/K_space_{k_type}.txt', index=False, sep=',')