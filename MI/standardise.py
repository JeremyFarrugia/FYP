import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd




ignore_indexes = [1,2] # Baseline trials (eyes open, eyes closed)
both_indexes = [5, 6, 9, 10, 13, 14] # Trials where t1 corresponds to both fists (otherwise left fist) and t2 corresponds to both feet (otherwise right fist)

# NOTE: Annotations mean the following: 0/t0: rest, 1/t1: left fist, 2/t2: right fist (t1 and t2 vary as written above)

emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']  # Channels present in the Emotiv EPOC x EEG device
# We will use the 14 channels that are present in the Emotiv EPOC x EEG device

sc_dir = os.path.dirname(__file__)

parent_dir = os.path.dirname(sc_dir)

segments_dir = os.path.join(sc_dir, "segmented_data")
os.makedirs(segments_dir, exist_ok=True)

standardised_dir = os.path.join(sc_dir, "standardised_data")
os.makedirs(standardised_dir, exist_ok=True)

cleaned_dir = os.path.join(sc_dir, "cleaned_data")

subjects = [folder for folder in os.listdir(cleaned_dir) if os.path.isdir(os.path.join(cleaned_dir, folder))]

labels = ['left_fist', 'right_fist', 'both_fists', 'both_feet', 'rest']

v1 = False

for subject_code in subjects:
    print(f"Cleaning data for subject {subject_code}")
    folder = os.path.join(cleaned_dir, subject_code)

    # Calculate the means and stds for each channel for the subject
    all_data = []

    for trial_num in range(3, 15): # 3 to 14 inclusive 
        file_name = f'{subject_code}R{trial_num:02d}.fif'   # E.g. S001R03.edf
        file_path = os.path.join(folder, file_name)

        if not os.path.exists(file_path):
            continue

        # Load data only for the 14 channels present in the Emotiv EPOC x EEG device
        raw = mne.io.read_raw_fif(file_path, preload=True)
        all_data.append(raw.get_data())

    all_data = np.concatenate(all_data, axis=1) # Concatenate all the data for the subject along the time axis
    means = np.mean(all_data, axis=1, keepdims=True)
    stds = np.std(all_data, axis=1, keepdims=True)

    # Segment and standardise the data
    for trial_num in range(3, 15):
        file_name = f'{subject_code}R{trial_num:02d}.fif'
        file_path = os.path.join(folder, file_name)

        if not os.path.exists(file_path):
            continue

        raw = mne.io.read_raw_fif(file_path, preload=True)
        if not v1:
            raw.plot(scalings=40e-6)
            plt.show()

        normalised_data = (raw.get_data() - means) / stds # Standardise the data (z-score normalisation)
        raw._data = normalised_data

        if not v1:
            raw.plot(scalings='auto')
            plt.show()
            v1 = True

        # Save the cleaned data to a new file
        file_name = f'{subject_code}R{trial_num:02d}_eeg.fif'
        save_path = os.path.join(standardised_dir, file_name)
        raw.save(save_path, overwrite=True)


