import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from autoreject import AutoReject
import json


sc_dir = os.path.dirname(__file__)
data_dir = sc_dir

standardised_dir = os.path.join(data_dir, "standardised")
os.makedirs(standardised_dir, exist_ok=True)

cleaned_dir = os.path.join(data_dir, "cleaned")

files = os.listdir(cleaned_dir)

def sliding_window(data: np.ndarray, window_size: int, step_size: int):
    """
    Create a sliding window over the data
    
    :param data: The data to create the sliding window over
    :param window_size: The size of the window (in samples - t*sfreq)
    :param step_size: The step size of the window (in samples - t*sfreq)
    :return: A generator that yields the windows
    """
    print(f"Data shape: {data.shape}")
    windows = []
    for i in range(0, data.shape[1] - window_size, step_size):
        window = data[:, i:i+window_size]

        windows.append(window)

    return np.array(windows)

event_label_mapping = {
    "A": "Apple",
    "C": "Car",
    "F": "Flower",
    "P": "Face"
}

tmin = 0
tmax = 4
sfreq = 128
ar = AutoReject(n_interpolate=[1, 2, 4], consensus=[0.85], cv=10, random_state=42, verbose=False, n_jobs=-1) # TODO: Unsure whether consensus of 0.9 gives better results, it seems to be close

segment_counts = []
montage = mne.channels.make_standard_montage('standard_1020')
# read first file
"""raw = mne.io.read_raw_fif(os.path.join(cleaned_dir, files[0]), preload=True)

raw.plot(scalings='auto')
plt.show()"""

# Organise files so we iterate over subjects

subject_data = {}
for file in files:
    subject_id = file.split("_")[0]
    if subject_id not in subject_data:
        subject_data[subject_id] = []
    subject_data[subject_id].append(file)

for subject_id, files in subject_data.items():
    #class_windows = {}
    segmented_fixed_data = []
    epoch_lengths = []
    classes = []
    for file in files:
        raw = mne.io.read_raw_fif(os.path.join(cleaned_dir, file), preload=True, verbose=False)
        raw.set_montage(montage)

        event_label = event_label_mapping[file.split("_")[1][0]]
        classes.append(event_label)

        # Segment the raw data into 4s sliding windows with 2s overlap
        segmented_data = []
        windows = sliding_window(raw.get_data(), window_size=int(sfreq * 4), step_size=int(sfreq * 2))
        print("Data shape:", windows.shape)
        print(f"Created {windows.shape[0]} segments from '{event_label}'")
        segmented_data = mne.EpochsArray(windows, raw.info, tmin=0, verbose=False)
        fixed_segments = ar.fit_transform(segmented_data)
        fixed_segments = fixed_segments.get_data()
        print(f"{fixed_segments.shape[0]} segments fixed and kept for label '{event_label}'")

        segmented_fixed_data.append(fixed_segments)
        epoch_lengths.append(fixed_segments.shape[0])

        #class_windows[event_label] = windows

    segment_counts.append(epoch_lengths)

    # Concatenate the fixed segments
    print(f"Data shape before concat 1: {len(segmented_fixed_data)}x{segmented_fixed_data[0].shape}")
    fixed_data = np.concatenate(segmented_fixed_data, axis=0)
    print(f"Data shape before concat 2: {fixed_data.shape}")
    fixed_data = np.concatenate(fixed_data, axis=1)
    means = np.mean(fixed_data, axis=1, keepdims=True)
    stds = np.std(fixed_data, axis=1, keepdims=True)

    fixed_data = (fixed_data - means) / stds
    fixed_raw = mne.io.RawArray(fixed_data, raw.info)

    

    annotations = raw.annotations
    # Save annotation descriptions
    annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=None)

    epoch_start = 0
    for i in range(len(classes)): # Annotate each event
        subsegment_count = epoch_lengths[i]
        print(f"Subsegment count for '{classes[i]}': {subsegment_count}")
        
        for j in range(subsegment_count):
            annotations.append(epoch_start + j * 4, 4, classes[i])
        epoch_start += subsegment_count * 4 # Each subsegment is 4s long
    fixed_raw.set_annotations(annotations)

    file = f"{subject_id}_eeg.fif"
    fixed_raw.save(os.path.join(standardised_dir, file), overwrite=True)
