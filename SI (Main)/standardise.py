import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from autoreject import AutoReject
import json


sc_dir = os.path.dirname(__file__)

standardised_dir = os.path.join(sc_dir, "segmented")
os.makedirs(standardised_dir, exist_ok=True)

cleaned_dir = os.path.join(sc_dir, "EEGDataCleaned")

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
tmin = 0
tmax = 40
sfreq = 128
ar = AutoReject(n_interpolate=[1, 2, 4], consensus=[0.85], cv=10, random_state=42, verbose=False, n_jobs=-1) # TODO: Unsure whether consensus of 0.9 gives better results, it seems to be close

segment_size = 4 # seconds
step_size = segment_size - 2 # segment size - overlap

segment_counts = []

for file in files:
    raw = mne.io.read_raw_fif(os.path.join(cleaned_dir, file), preload=True)

    # Segment into epochs depending on the markers
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)

    segmented_fixed_data = []
    epoch_lengths = []

    for event_label, _ in event_id.items():
        event_epochs = epochs[event_label]
        # Create a sliding window over the data
        segmented_data = []
        for epoch in event_epochs:
            segmented_data.extend(sliding_window(epoch, window_size=int(sfreq * segment_size), step_size=int(sfreq * step_size))) # 4s window with 2s overlap
        segmented_data = np.array(segmented_data)
        #segmented_data = sliding_window(event_epochs.get_data(), window_size=int(sfreq * 4), step_size=int(sfreq * 2)) # 4s window with 2s overlap
        #print(f"Created {segmented_data.shape[0]} segments from {event_label}")
        #print(f"Data shape: {segmented_data.shape}")
        # Create MNE epochs object from the segmented data
        segmented_data = mne.EpochsArray(segmented_data, event_epochs.info, tmin=0, verbose=False)
        fixed_segments = ar.fit_transform(segmented_data)
        fixed_segments = fixed_segments.get_data()
        print(f"{fixed_segments.shape[0]} segments fixed and kept for word '{event_label}'")

        # Convert from MNE epochs object to numpy array
        segmented_fixed_data.append(fixed_segments)
        epoch_lengths.append(fixed_segments.shape[0])

    segment_counts.append(epoch_lengths)

    # Concatenate the fixed segments
    #print(f"Segmented fixed data shape: {np.concatenate(segmented_fixed_data, axis=0).shape}")
    fixed_data = np.concatenate(segmented_fixed_data, axis=0)
    #concatanate dim 1 across all segments
    fixed_data = np.concatenate(fixed_data, axis=1)
    #print(f"Fixed data shape: {fixed_data.shape}")
    means = np.mean(fixed_data, axis=1, keepdims=True)
    stds = np.std(fixed_data, axis=1, keepdims=True)

    # Standardise the data
    fixed_data = (fixed_data - means) / stds

    # Create MNE raw object from fixed data
    fixed_raw = mne.io.RawArray(fixed_data, raw.info)
    
    # Create new annotations

    annotations = raw.annotations
    # Save annotation descriptions
    descriptions = annotations.description
    annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=None)

    epoch_start = 0
    for i in range(0, 20):
        subsegment_count = epoch_lengths[i]
        print(f"Subsegment count for '{descriptions[i]}': {subsegment_count}")
        
        for j in range(subsegment_count):
            annotations.append(epoch_start + j * segment_size, segment_size, descriptions[i])
        epoch_start += subsegment_count * segment_size # Each subsegment is 4s long
    fixed_raw.set_annotations(annotations)


    fixed_raw.save(os.path.join(standardised_dir, file), overwrite=True)
    