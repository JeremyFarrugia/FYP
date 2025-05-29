import mne
import os
from matplotlib import pyplot as plt
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), "VEP-EDF")
class_dirs = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

cleaned_folder = os.path.join(os.path.dirname(__file__), "cleaned")
os.makedirs(cleaned_folder, exist_ok=True)

print(class_dirs)

error_files = []


for class_folder in class_dirs:
    class_path = os.path.join(data_dir, class_folder)
    subfolders = [folder for folder in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, folder))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(class_path, subfolder)
        files = [file for file in os.listdir(subfolder_path) if file.endswith('.edf')]

        for file in files:
            file_path = os.path.join(subfolder_path, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                raw = raw.pick(emotiv_channels)
            except:
                print("Error reading", file_path)
                error_files.append(file)
                continue
                """# Get data from the csv instead
                print(f"Error reading {file_path}")
                csv_file_path = file_path.replace("VEP-EDF", "VEP-CSV").replace(".edf", ".csv")
                data = pd.read_csv(csv_file_path)
                special_mapping = {'EEG.'+channel_name: channel_name for channel_name in emotiv_channels}
                data = data.rename(columns=special_mapping)
                data = data[emotiv_channels]
                
                ch_types = ['eeg'] * len(emotiv_channels)
                ch_names = emotiv_channels
                sfreq = 128
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                raw = mne.io.RawArray(data.T, info)
                
                raw.plot(scalings='8e-5')
                plt.show()"""
            raw.interpolate_bads(reset_bads=True)
            raw.resample(128)
            raw.filter(2, 50)

            ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=800)

            ica.fit(raw)
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4'])


            ica.exclude = eog_indices

            raw = ica.apply(raw)

            raw.save(os.path.join(cleaned_folder, file.replace('.edf', '_eeg.fif')), overwrite=True)
            #raw.save(file_path.replace('.edf', '.fif'), overwrite=True)

print(error_files)
