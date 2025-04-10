import os
import pandas as pd

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
            with open(pkl_path, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = {}

        # Update with new values
        existing_data.update(dict_outputs)

        # Save the updated version
        with open(pkl_path, 'wb') as f:
            pickle.dump(existing_data, f)


