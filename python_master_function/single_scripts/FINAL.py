import os
import json
import glob
import pandas as pd
import mne
import numpy as np
import pickle
import argparse

from utils.saving import update_results_table, path_to_names
from features_EOC import features_EOC
from features_EOS import features_EOS
from features_Pred import features_Pred
from features_slope import features_slope
from features_DFA import features_DFA
from features_AVC import features_AVC
from features_avc_std_dist import features_avc_std_dist

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ========== CONFIG ========== #
MAX_TRIALS = 200
MAX_S = 2000


# Flags to enable/disable analyses
RUN_EOC = False
RUN_EOS = False
RUN_PRED = True
RUN_SLOPE = False
RUN_DFA = False
RUN_AVC = False
RUN_STD_DIST = False


# RUN_EOC = True
# RUN_EOS = True
# RUN_PRED = True
# RUN_SLOPE = True
# RUN_DFA = True
# RUN_AVC = True
# RUN_STD_DIST = True

if __name__ == "__main__":
    print("script started")
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
        print (paths)

    start = 0
    for path_i_relative, path in enumerate(paths[start:]):
        path_i = path_i_relative+start
   
        print(f"\n>>> Processing {os.path.basename(path)}")
        
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
        mne_epochs_32 = mne_epochs_raw.copy()
        mne_epochs_raw.pick('eeg')

        if len(mne_epochs_32) < 5 or len(mne_epochs_32.info['ch_names']) <  10:
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
                dict_data['pred_results_interpoalted'] = vals[11]
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


        # SAVE RESULTS
        update_results_table(path, row_data, results_table_path, results_dict_dir, dict_outputs=dict_data)
