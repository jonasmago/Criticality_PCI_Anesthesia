import os

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
