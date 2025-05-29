import mne
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Union


sc_dir = os.path.dirname(__file__)
data_dir = os.path.join(sc_dir, "standardised_data")

FREQ = 128
TIME = 4



def load_data(class2lab: dict[str: int], end: Union[None, int] = None, ignore_class: Union[None, list[int]] = None, ignore_subject: Union[None, list[int]] = None) -> tuple[list[np.ndarray], list[int], list[int]]:
    """
    Load the data from the standardised_data folder and return the inputs and targets as lists

    :param end: The number of files to load. If None, all files will be loaded
    :param ignore_class: A list of classes to ignore
    :param ignore_subject: A list of subjects to ignore
    :return: The inputs (List of numpy arrays), the targets (List of integers), the subject indexes (List of integers)
    """

    both_indexes = [5, 6, 9, 10, 13, 14] # Trials where t1 corresponds to both fists (otherwise left fist) and t2 corresponds to both feet (otherwise right fist)
    labels = ['left_fist', 'right_fist', 'both_fists', 'both_feet', 'rest']

    sc_dir = os.path.dirname(__file__)
    parent_dir = sc_dir
    #data_dir = os.path.join(parent_dir, "segmented_data") # Folder contraining the standardised segmented data
    data_dir = os.path.join(parent_dir, "standardised_data") # Folder contraining the standardised segmented data

    inputs = []
    targets = []
    subject_indexes = []

    channel_names = None

    tmin = 0 # Start at the onset of the trigger
    tmax = 4 # Take 4s of data after the trigger

    c = 0
    for file in tqdm(os.listdir(data_dir), desc='Loading data'):
        subject_index = int(file.split('S')[1][:3]) # Get the subject index from the filename

        if ignore_subject is not None and subject_index in ignore_subject:
            continue

        # Load .fif file
        raw = mne.io.read_raw_fif(os.path.join(data_dir, file), preload=True, verbose=False)

        if channel_names is None:
            # Reorder channel names to match predefined order
            """order = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
            # Reorder eeg data
            raw = raw.reorder_channels(order)"""
            channel_names = raw.ch_names
        
        

        # Segment the data by taking 4s of data after each trigger (from annotations)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)

        trial_num = int(file.split('R')[1][:2])

        if trial_num % 2 == 0:
            continue # Skip every second trial (Only consider imagined movements)



        for event_label, event_code in event_id.items():
            event_epochs = epochs[event_label]

            target_label: int = -1

            # Rename the label to be more descriptive
            if event_label == "T0":
                target_label = class2lab.get('rest', -1)
            elif event_label == "T1":
                if trial_num in both_indexes:
                    target_label = class2lab.get('both_fists', -1)
                else:
                    target_label = class2lab.get('left_fist', -1)
            elif event_label == "T2":
                if trial_num in both_indexes:
                    target_label = class2lab.get('both_feet', -1)
                else:
                    target_label = class2lab.get('right_fist', -1)
            else:
                raise ValueError(f"Invalid event label: {event_label}")

            if target_label == -1:
                continue # Skip this event

            if target_label > len(labels) or target_label < 0:
                raise ValueError(f"Invalid target label: {target_label}")
            
            if ignore_class is not None and target_label in ignore_class:
                continue # Skip this class

            for i, epoch in enumerate(event_epochs):
                inputs.append(epoch)
                targets.append(target_label)
                subject_indexes.append(subject_index)
        c+=1
        if end is not None and c >= end:
            break

    return inputs, targets, subject_indexes, channel_names

def channel_activation(channel_names: list[str], data: dict[str, np.ndarray], subject: Union[int, None], class_label: Union[int, None], squash: bool = True) -> None:
    """
    Create a topomap of the channel activation for a subject and class

    :param data: The data to use formated as a dictionary with 'inputs', 'targets' and 'subject'
    :param subject: The id of the subject to use, if None, all subjects will be used
    :param class_label: The class label to use, if None, all classes will be used
    """

    if subject is not None:
        mask = data['subject'] == subject
        print (mask)
        data = {key: data[key][mask] for key in data}

    if class_label is not None:
        mask = data['targets'] == class_label
        data = {key: data[key][mask] for key in data}

    print("Number of samples:", len(data['targets']))

    # Create a topompap from the mean of the data
    mean_data = np.mean(data['inputs'], axis=0)

    # Mean across time
    if squash:
        mean_data = np.mean(mean_data, axis=1, keepdims=True)

    info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
    evoked = mne.EvokedArray(mean_data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

    evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis')
    plt.show()

    plt.imshow(mean_data, aspect='auto', cmap='viridis')
    plt.yticks(range(len(channel_names)), channel_names)
    plt.show()
    
    # normalise mean_data
    mean_data = (mean_data - np.min(mean_data)) / (np.max(mean_data) - np.min(mean_data))
    plt.imshow(mean_data, aspect='auto', cmap='viridis')
    plt.yticks(range(len(channel_names)), channel_names)
    plt.show()

    print(mean_data)

def get_data(subject: int, class_label: int, data: dict[str, np.ndarray], squash: bool = True, normalise = True) -> np.ndarray:
    """
    Get the data for a specific subject and class

    :param subject: The subject index
    :param class_label: The class label
    :param data: The data to use formated as a dictionary with 'inputs', 'targets' and 'subject'
    :param squash: Whether to squash the data across time
    :param normalise: Whether to normalise the data
    :return: The data for the subject and class
    """

    if subject is not None:
        mask = data['subject'] == subject
        print (mask)
        data = {key: data[key][mask] for key in data}

    if class_label is not None:
        mask = data['targets'] == class_label
        data = {key: data[key][mask] for key in data}

    # Create a topompap from the mean of the data
    mean_data = np.mean(data['inputs'], axis=0)

    # Mean across time
    if squash:
        mean_data = np.mean(mean_data, axis=1, keepdims=True)

    if normalise:
        #mean_data = (mean_data - np.min(mean_data)) / (np.max(mean_data) - np.min(mean_data))
        mean_data = normalise_data(mean_data)

    return mean_data

def normalise_data(data: np.ndarray) -> np.ndarray:
    """
    Normalise the data to be between 0 and 1

    :param data: The data to normalise
    :return: The normalised data
    """

    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def channel_activation_all_subjects(channel_names: list[str], data: dict[str, np.ndarray], class_label: Union[int, None], squash: bool = True, normalise: bool = True) -> None:
    """
    Show the channel activation plot for each individual subject across all classes
    
    :param data: The data to use formated as a dictionary with 'inputs', 'targets' and 'subject'
    :param class_label: The class label to use, if None, all classes will be used
    """
    # Do the same with topomaps
    fig, axs = plt.subplots(4, 5, figsize=(20, 20))

    for i in range(data['subject'].max()):
        if i >= 20:
            break
        ax = axs[i // 5, i % 5]
        subject_data = get_data(i+1, class_label, data, squash, normalise=normalise)
        info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
        evoked = mne.EvokedArray(subject_data, info)

        montage = mne.channels.make_standard_montage('standard_1020')
        evoked.set_montage(montage)

        evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis', colorbar=False, axes=ax)
        ax.set_title(f"Subject {i+1}")

    # Get mean across all subjects
    mean_data = get_data(None, class_label, data, squash, normalise=normalise)
    ax = axs[-1, -1]
    info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
    evoked = mne.EvokedArray(mean_data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

    evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis', axes=ax, colorbar=False)
    ax.set_title("Mean")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(axs[-1, -1].images[0], cax=cbar_ax)


    # Hide the empty plots
    for i in range(data['subject'].max(), 20):
        ax = axs[i // 5, i % 5]
        ax.axis('off')

    plt.show()

def channel_activation_all_classes(channel_names: list[str], data: dict[str, np.ndarray], subject: Union[int, None], class2lab: dict, squash: bool = True, normalise: bool = True) -> None:
    """
    Show the channel activation plot for each individual class across all subjects
    
    :param data: The data to use formated as a dictionary with 'inputs', 'targets' and 'subject'
    :param subject: The subject index to use, if None, all subjects will be used
    """

    width_cm = 20
    width_inch = width_cm / 2.54
    height_inch = width_inch * (9/16)

    fig = plt.figure(figsize=(width_inch, height_inch), dpi=300)

    # Only display the overall mean
    mean_data = get_data(None, class_label, data, False, normalise=normalise)
    mean_data = mean_data.reshape(mean_data.shape[0], -1)  # Reshape to 2D
    print("Mean data shape:", mean_data.shape)
    ax = fig.add_subplot(111)
    ax.imshow(mean_data, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_xticks(np.arange(0, mean_data.shape[1], 512/4))
    ax.set_xticklabels(np.arange(0, mean_data.shape[1], 512/4) / FREQ, fontsize=9)  # Convert to seconds
    ax.set_ylabel("Channels", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)



    plt.show()

    # Do the same with topomaps
    fig, axs = plt.subplots(1, 5, figsize=(20, 20), dpi=300)

    for i in range(data['targets'].max()+1):
        ax = axs[i % 5]
        class_data = get_data(subject, i, data, squash, normalise=normalise)
        info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
        evoked = mne.EvokedArray(class_data, info)

        montage = mne.channels.make_standard_montage('standard_1020')
        evoked.set_montage(montage)

        evoked.plot_topomap(times=0, ch_type='eeg', show_names=False, outlines='head', show=False, cmap = 'viridis', colorbar=False, axes=ax)
        ax.set_title(f"{list(class2lab.keys())[list(class2lab.values()).index(i)]}")

    # Get mean across all classes
    mean_data = get_data(subject, None, data, squash, normalise=normalise)
    ax = axs[-1]
    info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
    evoked = mne.EvokedArray(mean_data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

    evoked.plot_topomap(times=0, ch_type='eeg', show_names=False, outlines='head', show=False, cmap = 'viridis', axes=ax, colorbar=False)
    ax.set_title("Average")
    
    # Plot the colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #plt.colorbar(axs[-1].images[0], cax=cbar_ax)

    fig.subplots_adjust(right=0.86, left=0.03, top=0.925, bottom=0.04)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = fig.add_axes([0.885, 0.35, 0.05, 0.3])
    plt.colorbar(axs[-1].images[0], cax=cbar_ax)


    # Hide the empty plots
    """for i in range(data['targets'].max()+1, 24):
        ax = axs[i // 6, i % 6]
        ax.axis('off')"""

    plt.show()

        


    

class2lab = {
    'left_fist': 0,
    'right_fist': 1,
    'both_fists': 2,
    'both_feet': 3
}

inputs, targets, subject_indexes, channel_names = load_data(class2lab, end=None, ignore_class=None, ignore_subject=None)

data = {
    'inputs': np.array(inputs),
    'targets': np.array(targets),
    'subject': np.array(subject_indexes)
}

class_label = None

#channel_activation(channel_names, data, subject=1, class_label=None, squash=True)
#channel_activation_all_subjects(channel_names, data, class_label=class_label, squash=True)
channel_activation_all_classes(channel_names, data, subject=None, class2lab=class2lab, squash=True, normalise=True)

