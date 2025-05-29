# calculate the snr of the EEG data

import numpy as np
import scipy.signal as signal
import mne
import os
from typing import Union
from matplotlib import pyplot as plt

def sliding_window(input: np.ndarray, freq: int, step_size: float, window_size: float) -> list[np.ndarray]:
    """
    Apply sliding window to input data

    :param input: The input data
    :param freq: The frequency the data was sampled at
    :param step_size: The step size in seconds for the sliding window
    :param window_size: The size in seconds of the window
    :return: The windowed data
    """
    step_size = int(freq * step_size)
    window_size = int(freq * window_size)

    windows = []

    thresh = 5
    for i in range(0, (input.shape[1] - window_size), step_size):
        window = input[:, i:i+window_size]

        windows.append(window)
    
    return windows

def data_power(data: np.ndarray, squeeze: bool = True) -> Union[np.ndarray, float]:
    """
    Calculate the power of the data

    :param data: The data to calculate the power for
    :param squeeze: Whether to squeeze the data
    :return: The power of the data
    """
    if squeeze:
        return np.mean(np.square(data))
    else:
        return np.mean(np.square(data), axis=1)

def get_snr_for_subject(file: str, squeeze: bool = True) -> dict:
    """
    Calculate the SNR for a subject

    :param file: The file to calculate the SNR for
    :return: The SNR for each word
    """
    raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    word_segments = {key : [] for key in event_id.keys()}
    pre_word_segments = {key : [] for key in event_id.keys()}
    #print(word_segments)


    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=40, baseline=None, preload=True, verbose=False)

    # Get pre epoch data (2s before the trigger)
    pre_epochs = mne.Epochs(raw, events, event_id, tmin=-2, tmax=0, baseline=None, preload=True, verbose=False)


    for event_label, _ in event_id.items():
        event_epochs = epochs[event_label]

        for i, epoch in enumerate(event_epochs):
            windows = sliding_window(epoch, freq=128, step_size=2, window_size=2)
            word_segments[event_label].extend(windows)
            
        for i, epoch in enumerate(pre_epochs[event_label]):
            pre_word_segments[event_label] = epoch
            if i > 0:
                raise ValueError("More than one pre-word segment found")
            

    word_snr = {}
    for word in word_segments.keys():
        pre_stimulus_data = np.array(pre_word_segments[word])
        
        noise_power = data_power(pre_stimulus_data, squeeze=squeeze)
        
        snr_for_word = []
        
        for i, stimulus_data in enumerate(word_segments[word]):
            signal_power = data_power(stimulus_data, squeeze=squeeze)
            
            # SNR between 2s pre-stimulus and 4s post-stimulus
            snr = 10 * np.log10(signal_power / noise_power)
            snr_for_word.append(snr)
            
            
        #print(f"SNR for individual segments:\n{snr_for_word}")
        word_snr[word] = np.mean(snr_for_word, axis=0)
        #print(f"Subject {subject_index} Word {word}: {snr_for_word} dB")
        
    return word_snr

def topomap(subject_snr: dict, channel_names: list[str], freq: int):
    """
    Plot the topomap for each subject

    :param subject_snr: The SNR for each subject
    :param channel_names: The channel names
    :param freq: The frequency the data was sampled at
    """

    fig, axs = plt.subplots(4, 5, figsize=(20, 20))

    for i, subject in enumerate(subject_snr.keys()):
        ax = axs[i // 5, i % 5]
        info = mne.create_info(ch_names=channel_names, sfreq=freq, ch_types='eeg')
        # Reformat subject data into a 14,1 array (14 channels, 1 sample)
        subject_data = np.expand_dims(subject_snr[subject], axis=1)
        evoked = mne.EvokedArray(subject_data, info)

        montage = mne.channels.make_standard_montage('standard_1020')
        evoked.set_montage(montage)

        evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis', colorbar=False, axes=ax)
        ax.set_title(f"Subject {i+1}")

    # Get mean across all subjects
    mean_data = np.mean(list(subject_snr.values()), axis=0)

    ax = axs[-1, -1]
    info = mne.create_info(ch_names=channel_names, sfreq=freq, ch_types='eeg')
    mean_data = np.expand_dims(mean_data, axis=1)
    evoked = mne.EvokedArray(mean_data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

    evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis', axes=ax, colorbar=False)
    ax.set_title("Mean")

    plt.show()

sc_dir = os.path.dirname(__file__)
data_dir = os.path.join(sc_dir, "EEGDataCleaned") #
avg = 0
subject_snr = {}
channel_names = None

squeeze = False                 # Whether to squeeze the data (average across all channels)
for file in os.listdir(data_dir):
    if not file.endswith(".fif"):
        continue
    
    if channel_names is None:
        raw = mne.io.read_raw_fif(os.path.join(data_dir, file), preload=True, verbose=False)
        channel_names = raw.ch_names

    subject_index = int(file.split("_")[0][1:])
    word_snr = get_snr_for_subject(os.path.join(data_dir, file), squeeze=squeeze)
    
    
    """print(f"SNR for each word for subject {subject_index}:\n")
    for word, snr in word_snr.items():
        print(f"{word}: {snr} dB")"""
        
    subject_snr[subject_index] = np.mean(list(word_snr.values()), axis=0)
    print("\n")
    print(f"Average SNR for participant {subject_index}: {subject_snr[subject_index]} dB")
    avg += subject_snr[subject_index]
    
print(f"Average SNR for all participants: {avg / len(os.listdir(data_dir))} dB")

if not squeeze:
    topomap(subject_snr, channel_names, 128)