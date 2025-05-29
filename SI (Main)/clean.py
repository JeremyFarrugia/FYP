import mne
import numpy as np
import matplotlib.pyplot as plt
import os

sc_dir = os.path.dirname(__file__)

clean_dir = os.path.join(sc_dir, "EEGDataCleaned")
os.makedirs(clean_dir, exist_ok=True)

raw_data_dir = os.path.join(sc_dir, "EEGDataFixed")

files = os.listdir(raw_data_dir)


for file in files:
    raw = mne.io.read_raw_fif(os.path.join(raw_data_dir, file), preload=True)

    # raw = raw.set_eeg_reference("average", projection=True).apply_proj() # Removed as it just flattens the data, likely removing distinguishable features

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)

    # Resample to 128 Hz
    raw.resample(128)

    # Band pass filter between 2 and 50 Hz
    raw.filter(2, 50)

    ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=800)

    ica.fit(raw)

    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4'])


    ica.exclude = eog_indices

    raw = ica.apply(raw, exclude=eog_indices)


    raw.save(os.path.join(clean_dir, file), overwrite=True)
