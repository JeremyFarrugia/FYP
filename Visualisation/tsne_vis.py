import os
import numpy as np
import torch
import EEGDeformer
from tqdm import tqdm
import mne
from torch.utils.data import DataLoader
from typing import Union
import optuna
import json
import random
import matplotlib.pyplot as plt
import einops
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import time
from captum.attr import IntegratedGradients
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

SEED = 56
FREQ = 128
TIME = 4

def load_data(class2lab: dict[str, int], end: Union[None, int] = None, ignore_class: Union[None, list[int]] = None, ignore_subject: Union[None, list[int]] = None) -> tuple[list[np.ndarray], list[int], list[int], list[str]]:
    """
    Load the data from the standardised_data folder and return the inputs and targets as lists

    :param class2lab: A dictionary mapping class names to labels
    :param end: The number of files to load. If None, all files will be loaded
    :param ignore_class: A list of class indexes to ignore
    :param ignore_subject: A list of subject indexes to ignore
    :return: The inputs (List of numpy arrays), the targets (List of integers), the subject indexes (List of integers) and the channel names (List of strings)
    """

    sc_dir = os.path.dirname(__file__)
    data_dir = os.path.join(sc_dir, "EEGDataStandardised") # Folder contraining the standardised data
    #data_dir = os.path.join(sc_dir, "segmented") # TODO change back
    #data_dir = os.path.join(sc_dir, "no_overlap")

    inputs = []
    targets = []
    subject_indexes = []

    channel_names = None # Channel names for the topomap

    tmin = 0 # Start at the onset of the trigger
    tmax = 40 # Take 40s of data after the trigger

    c = 0
    for file in tqdm(os.listdir(data_dir), desc='Loading data'):
        # Load .fif file
        raw = mne.io.read_raw_fif(os.path.join(data_dir, file), preload=True, verbose=False)

        if channel_names is None:
            channel_names = raw.ch_names # This gets updated each time but doesn't matter as all channel names should be the same

        # Segment the data by taking 4s of data after each trigger (from annotations)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        #epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=4, baseline=None, preload=True, verbose=False) # NOTE: I change tmax to 4s from 40s as I changed the way the data is segmented

        subject_index = int(file.split('_')[0][1:])

        if ignore_subject is not None and subject_index in ignore_subject:
            continue

        for event_label, _ in event_id.items():
            event_epochs = epochs[event_label]

            target_label: int = class2lab[event_label]

            if ignore_class is not None and target_label in ignore_class:
                continue

            

            
            for i, epoch in enumerate(event_epochs):

                if epoch.shape[1] != FREQ * TIME:
                    epoch = epoch[:, :FREQ * TIME] # Trim the data to 4s if it is longer

                inputs.append(epoch)
                targets.append(target_label)
                subject_indexes.append(subject_index)
        c+=1
        if end is not None and c >= end:
            break

    print(f"Loaded {len(inputs)} samples")

    # Standardise inputs to have mean 0 and std 1
    """inputs = np.array(inputs) # TODO : remove
    inputs = (inputs - np.mean(inputs, axis=1, keepdims=True)) / np.std(inputs, axis=1, keepdims=True)"""

    return inputs, targets, subject_indexes, channel_names

def tsne_plot(inputs: np.ndarray, targets: np.ndarray, idx: np.ndarray, ch_names: list[str], class2lab: dict[str, int], params: dict, model_path: str, perplexity: int = 30) -> None:

    inputs, targets, idx = np.array(inputs), np.array(targets), np.array(idx)
    # Filter to a specific participant
    participant = 1
    """mask = idx != participant
    inputs = inputs[mask]
    targets = targets[mask]
    idx = idx[mask]"""

    model = EEGDeformer.Deformer(
        num_chan=14,
        num_time=FREQ*TIME,
        temporal_kernel=params['temporal_kernel'],
        num_kernel=params['num_kernel'],
        num_classes= 4,
        depth=params['depth'],
        heads=params['heads'],
        mlp_dim=params['mlp_dim'],
        dim_head=params['dim_head'],
        dropout=params['dropout']
    )

    # Change head to match the number of classes

    model.load_state_dict(torch.load(model_path))
    model.change_head(num_classes=len(class2lab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for input, target in tqdm(zip(inputs, targets), desc="Extracting features"):
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(input)
            features.append(output.cpu().numpy())
            labels.append(target)

    X_feat = np.concatenate(features, axis=0)
    y = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    X_embedded = tsne.fit_transform(X_feat)

    lab2class = {v: k for k, v in class2lab.items()}

    """plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab20')
    plt.colorbar()
    plt.title("t-SNE visualization of EEG data")
    plt.show()"""

    categories = np.unique(y)

    num_categories = len(categories)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)

    category_names = [lab2class[cat] for cat in categories]

    dot_width = 2
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab20', s=dot_width)
    cbar = plt.colorbar(scatter, ticks=categories) 
    cbar.ax.set_yticklabels(category_names)

    plt.show()


    

    # Flatten inputs so each data point is 1D
    """inputs_flattened = [input.flatten() for input in inputs]
    inputs_flattened = np.array(inputs_flattened)

    # Perform PCA to reduce dimensionality to 50 components
    pca = PCA(n_components=10)
    inputs_pca = pca.fit_transform(inputs_flattened)
    

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(inputs_pca)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=idx, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE visualization of EEG data")
    plt.show()"""


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    class2lab = {
        "I":1,
        "More":2,
        "What":3,
        "Who":4,
        "Help":5,
        "You":6,
        "Want":7,
        "When":8,
        "On":9,
        "Stop":10,
        "Yes":11,
        "Do":12,
        "Where":13,
        "In":14,
        "Give":15,
        "No":16,
        "Put":17,
        "Why":18,
        "Get":19,
        "Come":20,
    }

    # Set class2lab keys to lowercase
    class2lab = {key.lower(): value for key, value in class2lab.items()}
    # -1 from values to make them 0-indexed
    class2lab = {key: value-1 for key, value in class2lab.items()}


    inputs, targets, idx, ch_names = load_data(class2lab, ignore_class=None, ignore_subject=None)

    study_db_path = os.path.join(os.path.dirname(__file__), "deformer_study.db")
    study3 = optuna.create_study(direction='maximize', study_name='quick_tune', storage='sqlite:///{}'.format(study_db_path), load_if_exists=True)
    best_params = study3.best_params

    best_params = {'lr': 0.005, 'batch_size': 32, 'dropout': 0.25, 'temporal_kernel': 11, 'num_kernel': 128, 'depth': 6, 'heads': 12, 'mlp_dim': 24, 'dim_head': 64}


    #test = "baseline all subjects tuned"
    #test = "MI all subjects"
    #test = "VI all subjects"


    #model_folder = os.path.join(os.path.dirname(__file__), "test", test)
    #model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pth')]

    #model_path = model_paths[0] if len(model_paths) > 0 else None
    #if model_path is None:
    #    raise ValueError(f"No model found in {model_folder}")
    #print(f"Loading model from {model_path}")

    model_path = os.path.join(os.path.dirname(__file__), "deformer_base_VI.pth")


    tsne_plot(inputs, targets, idx, ch_names, class2lab, best_params, model_path=model_path, perplexity=30)

if __name__ == "__main__":
    main()