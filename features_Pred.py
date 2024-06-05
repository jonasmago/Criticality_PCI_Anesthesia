#!/usr/bin/env python
import pandas as pd
from antropy import lziv_complexity
import multiprocessing as mp
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import argparse
import mne.io
import random
import mne
import os
import multiprocessing as mp
import neurokit2 as nk
from scipy.fft import fft, ifft
from scipy.signal import hilbert
from scipy.fftpack import rfft, irfft

# call:   python features_Pred.py -data_dir EPOCHS -output_dir RESULTS -part_info EPOCHS/participants.txt -lfrequ 0.1 -hfrequ 40

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

    print('Done Trial {}'.format(str(trial)))

    return Lyaps, Dims, Ent, LZC, KDF


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Complexity using different methods')
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


    # make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load patient info and conditions
    info = pd.read_csv(args.part_info,sep = ',', index_col=None)
    P_IDS = info['ID']
    Cond = info['Cond']
    Drug = info['Drug']

    Lyaps_max = []
    Lyaps_space = []
    Dims_mean = []
    Dims_space = []
    Ent_mean = []
    Ent_space = []
    LZC_mean = []
    LZC_space = []
    KDF_mean = []
    KDF_space = []

    #loop over all conditions and particiants
    for i, p_id in enumerate(P_IDS):
        print(f"Analyzing Lyapunov Exponenf of {p_id}");

        #################################
        #          LOAD  DATA          #
        #################################

        input_fname = f"{in_dir}/epochs_{Drug[i]}_{Cond[i]}_{P_IDS[i]}.mat"

        data = loadmat(input_fname)
        epochs = data['trails']
        fs = epochs[0].shape[1]/10

        info = mne.create_info(ch_names=60, sfreq=fs, ch_types='eeg')
        epochs_mne = mne.EpochsArray(epochs, info)

        # prepare data
        epochs_res = epochs_mne.resample(250)
        epochs_filt = epochs_res.filter(lfreq, hfreq, verbose = False)

        # if data is too long only use the first 3 min of data
        nr_trials = min([len(epochs_filt),30])
        epochs = epochs_filt.get_data()
        nr_channels =  epochs.shape[1]

        #################################
        #    Calculate LZC             #
        #################################

        pool = mp.Pool(mp.cpu_count())

        # loop over every time segment
        input = []
        for trial in range(nr_trials):
            input.append((trial,epochs))

        results = pool.starmap(get_pred,input)

        Lyaps_max.append(np.mean(np.array(results)[:,0,:],axis = (0,1)))
        Lyaps_space.append(np.mean(np.array(results)[:,0,:],axis = (0)))
        Dims_mean.append(np.mean(np.array(results)[:,1,:],axis = (0,1)))
        Dims_space.append(np.mean(np.array(results)[:,1,:],axis = (0)))
        Ent_mean.append(np.mean(np.array(results)[:,2,:],axis = (0,1)))
        Ent_space.append(np.mean(np.array(results)[:,2,:],axis = (0)))
        LZC_mean.append(np.mean(np.array(results)[:,3,:],axis = (0,1)))
        LZC_space.append(np.mean(np.array(results)[:,3,:],axis = (0)))
        KDF_mean.append(np.mean(np.array(results)[:,4,:],axis = (0,1)))
        KDF_space.append(np.mean(np.array(results)[:,4,:],axis = (0)))

        pool.close()

        # save part
        output_df = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1],
                    'Lyaps_max':Lyaps_max, 'Dims_mean':Dims_mean,
                    'Ent_mean':Ent_mean, 'LZC_mean':LZC_mean,'KDF_mean':KDF_mean }
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{out_dir}/Pred_{lfreq}_{hfreq}.txt', index=False, sep=',')

        # save space
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(Lyaps_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/Lyaps_space_{lfreq}_{hfreq}.txt', index=False, sep=',')

        # save space
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(Dims_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/Dims_space_{lfreq}_{hfreq}.txt', index=False, sep=',')

        # save space
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(Ent_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/Ent_space_{lfreq}_{hfreq}.txt', index=False, sep=',')

        # save space
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(LZC_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/LZC_space_{lfreq}_{hfreq}.txt', index=False, sep=',')

        # save space
        output_df_space = {'ID':P_IDS[0:i+1], 'Drug': Drug[0:i+1],'Cond':Cond[0:i+1]}
        output_df = pd.concat((pd.DataFrame(output_df_space), pd.DataFrame(KDF_space).reset_index(drop=True)), axis = 1)
        output_df.to_csv(f'{out_dir}/KDF_space_{lfreq}_{hfreq}.txt', index=False, sep=',')
