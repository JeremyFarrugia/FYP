import mne
import numpy as np
import matplotlib.pyplot as plt
import os



ignore_indexes = [1,2] # Baseline trials (eyes open, eyes closed)
both_indexes = [5, 6, 9, 10, 13, 14] # Trials where t1 corresponds to both fists (otherwise left fist) and t2 corresponds to both feet (otherwise right fist)

emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']  # Channels present in the Emotiv EPOC x EEG device
# We will use the 14 channels that are present in the Emotiv EPOC x EEG device

sc_dir = os.path.dirname(__file__)

clean_dir = os.path.join(sc_dir, "cleaned_data")

raw_data_dir = os.path.join(sc_dir, "raw_data")

subjects = [folder for folder in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, folder))]

#subject_code = 'S001'

for subject_code in subjects:
    print(f"Cleaning data for subject {subject_code}")
    folder = os.path.join(raw_data_dir, subject_code)
    for trial_num in range(3, 15): # 3 to 14 inclusive 
        file_name = f'{subject_code}R{trial_num:02d}.edf'   # E.g. S001R03.edf
        file_path = os.path.join(folder, file_name)

        # Load data only for the 14 channels present in the Emotiv EPOC x EEG device
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Compare channels and select it if the channel name has the same 'pattern' (lowercase emotiv channel exists in the channel string)
        channel_names = []
        name_map = {} # This will be used to rename the channels to a standard name (Just some convenience)
        for channel in raw.ch_names: # weird way to do it, but the dataset creators used weird channel names, including . in 
            channel_c = channel.replace(".", "")
            for emotiv_channel in emotiv_channels:
                if emotiv_channel.lower() == channel_c.lower():
                    channel_names.append(channel)
                    name_map[channel] = emotiv_channel
                    break

        # Pick only the channels that are present in the Emotiv EPOC x EEG device
        raw.pick_channels(channel_names)
        # Rename the channels to a standard name
        raw.rename_channels(name_map)
        # Resample to 128 Hz
        raw.resample(128)

        # Band pass filter between 2 and 50 Hz
        raw.filter(2, 50)

        # Perform ICA on AF3 and AF4 channels
        ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=800)
        ica.fit(raw)

        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4'])


        ica.exclude = eog_indices

        raw = ica.apply(raw)
        # Save the cleaned data
        clean_file_name = f'{subject_code}R{trial_num:02d}.fif' # MNE saves the data in fif format
        clean_file_path = os.path.join(clean_dir, subject_code, clean_file_name)
        os.makedirs(os.path.dirname(clean_file_path), exist_ok=True)
        raw.save(clean_file_path, overwrite=True)