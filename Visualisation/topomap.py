# Draw a blank mne topomap only including emotiv channels
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import os
from mne.channels import make_standard_montage

emotiv_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

def draw_blank_topomap():
    # Create a standard montage
    montage = make_standard_montage('standard_1020')

    # Create an info structure with the emotiv channels
    info = mne.create_info(ch_names=emotiv_channels, sfreq=128, ch_types='eeg')
    info.set_montage(montage)

    # Create a blank raw object
    raw = mne.io.RawArray(data=[[]*14]*14, info=info)

    # Plot the topomap
    fig, ax = plt.subplots(dpi=300)
    mne.viz.plot_sensors(raw.info, show_names=True, axes=ax, kind='topomap', ch_type='eeg')
    plt.show()

if __name__ == "__main__":
    draw_blank_topomap()