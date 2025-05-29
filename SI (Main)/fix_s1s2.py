import mne
import numpy as np
import os
from matplotlib import pyplot as plt
import json

"""
s1 and s2 have all their markers (excluding the first) placed early (at the beginning of the rest of the previous word) and so need to be shifted to the right position
s1 is also missing its first marker "you" which is added in its correct position

Additionally, this script helps clean the data of all the EEG files, appropriately renaming markers and removing extra (non-EEG) channels
"""


sc_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sc_dir, 'EEGDataRaw')
eeg_dir = os.path.join(sc_dir, 'EEGDataFixed')

emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

os.makedirs(eeg_dir, exist_ok=True)

eeg_files = [f for f in os.listdir(data_dir) if f.endswith('.edf')] # S1 and s2 handled separately
print(eeg_files)

montage = mne.channels.make_standard_montage('standard_1020')

raw = mne.io.read_raw_edf(os.path.join(data_dir, 'S1_eeg.edf'), preload=True)
annotations = raw.annotations

"""print(f"Channels: {raw.ch_names}")
raw.plot(scalings='80e-6')
plt.show()"""


s1_marker1 = "you"
# create new annotation 40s before current first annotation
annotations.append(annotations.onset[0] - 40, 40, s1_marker1)

# Iterate through annotations, organising by order shown

for i in range(len(annotations)):
    if(annotations.description[i] == "you"): # Skip our custom marker
        continue
    annotations.description[i] = annotations.description[i].split(",")[0]
    annotations.onset[i] = annotations.onset[i] + 10 # Shift all annotations 10s forward to account for the delay in the markers
    annotations.duration[i] = 40 # Set all durations to 40s


# Copy s1 and fix markers
raw_s1 = raw.copy().pick(emotiv_channels)
raw_s1.set_annotations(annotations)

raw_s1.set_montage(montage)

# Save s1
raw_s1.save(os.path.join(eeg_dir, 'S1_eeg.fif'), overwrite=True)

# Fix s2 markers
raw = mne.io.read_raw_edf(os.path.join(data_dir, 'S2_eeg.edf'), preload=True)
annotations = raw.annotations

for i in range(len(annotations)):
    annotations.description[i] = annotations.description[i].split(",")[0]
    if i != 0:
        annotations.onset[i] = annotations.onset[i] + 7 # 7s rest for s2
    annotations.duration[i] = 40

# Copy s2 and fix markers
raw_s2 = raw.copy().pick(emotiv_channels)
raw_s2.set_annotations(annotations)

raw_s2.set_montage(montage)


raw_s2.save(os.path.join(eeg_dir, 'S2_eeg.fif'), overwrite=True)

# Fix s5 - missing initial marker ('on')
raw = mne.io.read_raw_edf(os.path.join(data_dir, 'S5_eeg.edf'), preload=True)
annotations = raw.annotations

w1_marker = "on"
annotations.append(annotations.onset[0] - 48, 40, w1_marker) # 50s earlier (before 8s rest and lasting 40s)

for i in range(len(annotations)):
    if annotations.description[i] == "on":
        continue
    annotations.description[i] = annotations.description[i].split(",")[0]
    annotations.duration[i] = 40

raw_s5 = raw.copy().pick(emotiv_channels)
raw_s5.set_annotations(annotations)

raw_s5.set_montage(montage)

# Mark p7 as bad channel
raw_s5.info['bads'].append('P7')

raw_s5.save(os.path.join(eeg_dir, 'S5_eeg.fif'), overwrite=True)


# Fix s5 - missing initial marker ('help')
raw = mne.io.read_raw_edf(os.path.join(data_dir, 'S21_eeg.edf'), preload=True)
annotations = raw.annotations

w1_marker = "help"
annotations.append(annotations.onset[0] - 50, 40, w1_marker) # 50s earlier (before 8s rest, 2s warning and lasting 40s)

for i in range(len(annotations)):
    if annotations.description[i] == "help":
        continue
    annotations.description[i] = annotations.description[i].split(",")[0]
    annotations.duration[i] = 40

raw_s21 = raw.copy().pick(emotiv_channels)
raw_s21.set_annotations(annotations)

raw_s21.set_montage(montage)

raw_s21.plot(scalings=80e-6)
plt.show()

raw_s21.save(os.path.join(eeg_dir, 'S21_eeg.fif'), overwrite=True)


ignore = ['s1','s2','s5','s21'] # Ignore s1,s2 and s5 as they have been handled separately


bad_channels = { # T7 was problematic for many participants, with some I managed to adequately adjust, others were too poor
    's5': ['O1'],
    's11': ['T7'],
    's12': ['O1'],
    's13': ['T7'],
    's14': ['T7'],
}

# Clean all other files
for file in eeg_files:
    participant = file.split("_")[0].lower()
    if participant in ignore: # Create montage for ignored participants as they haven't beend done yet
        continue

    """if participant == 's6':
        raw.plot(scalings='80e-6')
        plt.show()"""

    raw = mne.io.read_raw_edf(os.path.join(data_dir, file), preload=True)
    annotations = raw.annotations
    raw = raw.pick(emotiv_channels)

    for i in range(len(annotations)):
        annotations.description[i] = annotations.description[i].split(",")[0]
        annotations.duration[i] = 40

    if participant in bad_channels:
        raw.info['bads'] = bad_channels[participant]

    raw.set_annotations(annotations)
    raw.set_montage(montage) # Set montage (Electrode positions)
    raw.save(os.path.join(eeg_dir, file.replace(".edf", ".fif")), overwrite=True)

    """raw.plot(scalings='auto')
    plt.show()"""

# Show electrode positions
"""raw.get_montage().plot()
plt.show()"""

