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

SEED = 56
FREQ = 128
TIME = 4

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_in_s = end - start
        time_in_m = time_in_s // 60
        time_in_s = time_in_s % 60
        time_in_h = int(time_in_m // 60)
        time_in_m = int(time_in_m % 60)
        print(f"Training took {time_in_h}h {time_in_m}m {time_in_s:.1f}s")
        return result
    return wrapper



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

    for i in range(0, (input.shape[1] - window_size), step_size):
        window = input[:, i:i+window_size]

        windows.append(window)
    
    return windows


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


def split_data(inputs: list[np.ndarray], targets: list[int], subject_indexes: list[int], split: float = 0.8, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the inputs and targets into training and validation sets (Train inputs, Train targets, Val inputs, Val targets)
    Ensures proportional representation of each class for each subject in both training and validation sets

    :param inputs: The inputs
    :param targets: The target labels for each input
    :param subject_indexes: The subject index for each input (Who the data was recorded from, used to ensure that subjects appear in both training and validation sets)
    :param split: The proportion of the data to use for training

    :return: The training inputs, the training targets, the training subject indexes, the validation inputs, the validation targets, the validation subject indexes
    """
    if shuffle:
        indices = np.arange(len(targets))
        np.random.shuffle(indices)
        inputs = np.array(inputs)[indices]
        targets = np.array(targets)[indices]
        subject_indexes = np.array(subject_indexes)[indices]

    # Generate a dictionary for the quotas for each class for each subject (To ensure proportional representation in training and validation sets)
    quotas = {}
    unique_subjects = np.unique(subject_indexes)
    for subject in unique_subjects:
        quotas[subject] = {label: 0 for label in np.unique(targets)}

    for i in range(len(inputs)):
        quotas[subject_indexes[i]][targets[i]] += 1

    # Adjust quotas to reflect the split
    for subject in unique_subjects:
        for label in quotas[subject]:
            quotas[subject][label] = int(quotas[subject][label] * split)

    train_inputs = []
    train_targets = []
    train_idx = []
    val_inputs = []
    val_targets = []
    val_idx = []

    # Allocate data according to quotas
    for i in range(len(inputs)):
        if quotas[subject_indexes[i]][targets[i]] > 0:
            train_inputs.append(inputs[i])
            train_targets.append(targets[i])
            train_idx.append(subject_indexes[i])
            quotas[subject_indexes[i]][targets[i]] -= 1
        else: # Once the training quota is satisfied, allocate the rest to the validation set
            val_inputs.append(inputs[i])
            val_targets.append(targets[i])
            val_idx.append(subject_indexes[i])

    

    return np.array(train_inputs), np.array(train_targets), np.array(train_idx), np.array(val_inputs), np.array(val_targets), np.array(val_idx)

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        if isinstance(inputs, np.ndarray):
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
        elif isinstance(inputs, torch.Tensor):
            self.inputs = inputs.clone().detach().float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.long)
        elif isinstance(targets, torch.Tensor):
            self.targets = targets.clone().detach().long()

    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        return {
            'inputs': self.inputs[idx],
            'targets': self.targets[idx]
        }
    
class EEGDataset_subjectID(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, subject_indexes):
        if isinstance(inputs, np.ndarray):
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
        elif isinstance(inputs, torch.Tensor):
            self.inputs = inputs.clone().detach().float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.long)
        elif isinstance(targets, torch.Tensor):
            self.targets = targets.clone().detach().long()

        if isinstance(subject_indexes, np.ndarray):
            self.subject_indexes = torch.tensor(subject_indexes, dtype=torch.long)
        elif isinstance(subject_indexes, torch.Tensor):
            self.subject_indexes = subject_indexes.clone().detach().long()

    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        return {
            'inputs': self.inputs[idx],
            'targets': self.targets[idx],
            'subjects': self.subject_indexes[idx]
        }
    
def get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=64) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation data loaders

    :param train_inputs: The training inputs
    :param train_targets: The training targets
    :param val_inputs: The validation inputs
    :param val_targets: The validation targets
    :param batch_size: The batch size
    """
    train_loader = DataLoader(EEGDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_inputs, val_targets), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

@time_wrapper
def train_model(model: torch.nn.Module, device: str, train_loader: DataLoader, val_loader: DataLoader, lr: float = 1e-4, lr_dropoff_factor: float = 0.5, max_epochs: int = 1000, patience_start: int = 20, label_smoothing: float = 0.1, save_path: str = None, interval: int = 1) -> tuple[list[float], list[float]]:
    """
    Train Deformer model, performing early stopping if the validation loss does not decrease for `patience` epochs

    :param model: The model to train
    :param train_loader: The training data loader
    :param val_loader: The validation data loader
    :param lr: The learning rate
    :param lr_dropoff_factor: The factor to drop the learning rate by after no improvement for 15 epochs
    :param max_epochs: The maximum number of epochs
    :param patience_start: The number of epochs to wait before early stopping
    :param label_smoothing: The label smoothing factor
    :param save_path: The path to save the model
    :param interval: The interval to output the training and validation loss
    :return: The training losses and validation losses
    """
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_dropoff_factor, patience=10, verbose=True) # Reduce learning rate on plateau
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_dropoff_factor, patience=20, verbose=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6) # Cosine annealing with restarts
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing) # Label smoothing to prevent overfitting

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    min_val_loss = float('inf')
    max_val_acc = 0

    patience = patience_start
    patience2 = patience_start * 2
    early_stop = False

    for epoch in range(1, max_epochs+1):
        model.train()
        val_loss = []
        val_acc = []
        val_f1 = []
        train_loss = []
        val_batch_sizes = []
        train_batch_sizes = []
        for batch in train_loader:

            optimizer.zero_grad()
            outputs = model(batch['inputs'].to(device))
            loss = criterion(outputs, batch['targets'].to(device))
            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_loss.append(loss)
            train_batch_sizes.append(len(batch['targets']))

        train_losses.append(np.average(train_loss, weights=train_batch_sizes))



        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['inputs'].to(device))
                loss = criterion(outputs, batch['targets'].to(device))
                loss = loss.item()
                val_acc.append((torch.argmax(outputs, dim=1) == batch['targets'].to(device)).sum().item() / len(batch['targets']))
                # Calculate F1 score
                val_f1.append(f1_score(batch['targets'].cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted'))
                val_loss.append(loss)
                val_batch_sizes.append(len(batch['targets']))

        val_loss = np.average(val_loss, weights=val_batch_sizes)
        val_losses.append(val_loss)
        val_accs.append(np.average(val_acc, weights=val_batch_sizes))
        val_f1s.append(np.average(val_f1, weights=val_batch_sizes))

        scheduler.step(val_loss) # Step the scheduler

        if val_loss > min_val_loss:
            patience -= 1
            if patience < 0:
                print("Early stopping")
                early_stop = True
                break
        else:
            patience = patience_start
            min_val_loss = val_loss
            # Save model checkpoint to revert to if performance degrades
            torch.save(model.state_dict(), "model_checkpoint.pth")

        if val_accs[-1] < max_val_acc:
            if max_val_acc - val_accs[-1] > 0.01:
                patience2 -= 1
            if patience2 < 0:
                print("Early stopping - Improvement of less than 1% accuracy in the last 50 epochs")
                early_stop = True
                break
        else:
            patience2 = patience_start * 2
            max_val_acc = val_accs[-1]

        if epoch % interval == 0:
            print(f"Epoch {epoch}/{max_epochs}\tTrain Loss: {train_losses[-1]:0.4f}\tVal Loss: {val_loss:0.4f}\tVal Acc: {np.average(val_acc):0.4f}\tVal F1: {np.average(val_f1):0.4f}")

    if early_stop:
        model.load_state_dict(torch.load("model_checkpoint.pth"))
    else:
        print("Training complete")

    os.remove("model_checkpoint.pth")

    # Save the model
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, val_accs, val_f1s

def load_base_model(model: torch.nn.Module, path: str, num_classes: int) -> torch.nn.Module:
    """
    Load a pre-trained model and change the head to predict `num_classes` classes

    :param model: The model to load
    :param path: The path to the pre-trained model
    :param num_classes: The number of classes to predict
    :return: The model with the new head
    """
    model.load_state_dict(torch.load(path))
    model.change_head(num_classes)
    return model

@time_wrapper
def fine_tune(model: torch.nn.Module, device: str, train_loader: DataLoader, val_loader: DataLoader, base_lr: float = 1e-4, head_lr: float = 1e-3, max_epochs: int = 1000, patience_start: int = 20, save_path: str = None, interval: int = 1) -> tuple[list[float], list[float]]:
    """
    Fine-tune the model, training only the head of the model

    :param model: The model to fine-tune
    :param train_loader: The training data loader
    :param val_loader: The validation data loader
    :param base_lr: The learning rate for the base layers
    :param head_lr: The learning rate for the head
    :param max_epochs: The maximum number of epochs
    :param patience: The number of epochs to wait before early stopping
    """
    model.train()

    # Separate parameters for the base and head of the model
    base_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'mlp_head' in name:
            head_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': base_lr},
        {'params': head_params, 'lr': head_lr}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.65, patience=10, verbose=True) # Reduce learning rate on plateau
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to prevent overfitting

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    min_val_loss = float('inf')

    patience = patience_start
    early_stop = False

    for epoch in range(1, max_epochs+1):
        model.train()
        val_loss = []
        val_acc = []
        val_f1 = []
        train_loss = []
        val_batch_sizes = []
        train_batch_sizes = []
        for batch in train_loader:

            optimizer.zero_grad()
            outputs = model(batch['inputs'].to(device))
            loss = criterion(outputs, batch['targets'].to(device))
            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_loss.append(loss)
            train_batch_sizes.append(len(batch['targets']))

        train_losses.append(np.average(train_loss, weights=train_batch_sizes))


        # Examine performance on validation set
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['inputs'].to(device))
                loss = criterion(outputs, batch['targets'].to(device))
                loss = loss.item()
                val_acc.append((torch.argmax(outputs, dim=1) == batch['targets'].to(device)).sum().item() / len(batch['targets']))
                # Calculate F1 score
                val_f1.append(f1_score(batch['targets'].cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted'))
                val_loss.append(loss)
                val_batch_sizes.append(len(batch['targets']))

        val_loss = np.average(val_loss, weights=val_batch_sizes)
        val_losses.append(val_loss)
        val_accs.append(np.average(val_acc, weights=val_batch_sizes))
        val_f1s.append(np.average(val_f1, weights=val_batch_sizes))

        scheduler.step(val_loss) # Step the scheduler


        


        if val_loss > min_val_loss:
            patience -= 1
            if patience < 0:
                print("Early stopping")
                early_stop = True
                break
        else:
            patience = patience_start
            min_val_loss = val_loss
            # Save model checkpoint to revert to if performance degrades
            torch.save(model.state_dict(), "model_checkpoint.pth")

        if epoch % interval == 0:
            print(f"Epoch {epoch}/{max_epochs}\tTrain Loss: {train_losses[-1]:0.4f}\tVal Loss: {val_loss:0.4f}\tVal Acc: {np.average(val_acc):0.4f}\tVal F1: {np.average(val_f1):0.4f}")

    if early_stop:
        model.load_state_dict(torch.load("model_checkpoint.pth"))
    else:
        print("Training complete")

    os.remove("model_checkpoint.pth")

    # Save the model
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, val_accs, val_f1s

def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: str) -> dict[str, list[int]]:
    """
    Evaluate the model on the validation set

    :param model: The model to evaluate
    :param val_loader: The validation data loader
    :return: The accuracy of the model on the validation set and the accuracy per class
    """
    model.eval()
    results = {'pred': [], 'true': [], 'subject': []}

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['inputs'].to(device))     # Get the model outputs
            _, predicted = torch.max(outputs, 1)            # Get the predicted classes

            results['pred'].extend(predicted.cpu().numpy())             # Store the predicted classes
            results['true'].extend(batch['targets'].cpu().numpy())      # Store the true classes
            results['subject'].extend(batch['subjects'].cpu().numpy())  # Store the subject indexes

    for i, (pred, true, subject) in enumerate(zip(results['pred'], results['true'], results['subject'])):
        results['pred'][i] = int(pred)
        results['true'][i] = int(true)
        results['subject'][i] = int(subject)


    
    return results

def stratify_data(inputs: list[np.ndarray], targets: list[int], subject_indexes: list[int]) -> dict[int, dict[int, list[np.ndarray]]]:
    """
    Stratify the data by subject and class

    :param inputs: The inputs
    :param targets: The targets
    :param subject_indexes: The subject indexes
    :return: A dictionary containing the data stratified by subject and class
    """

    stratified_data = {}

    unique_subjects = np.unique(subject_indexes)
    for subject in unique_subjects:
        stratified_data[subject] = {}

        mask = subject_indexes == subject
        subject_inputs = np.array(inputs)[mask]
        subject_targets = np.array(targets)[mask]

        unique_classes = np.unique(subject_targets)
        for label in unique_classes:
            mask = subject_targets == label
            stratified_data[subject][label] = subject_inputs[mask]

    return stratified_data


def create_folds(inputs, targets, subject_indexes, num_folds=5):
    """
    Create stratified folds ensuring both class and subject balance.
    """
    # Convert to NumPy arrays for indexing
    inputs, targets, subject_indexes = np.array(inputs), np.array(targets), np.array(subject_indexes)

    # Sort by subjects, then class
    sorted_indices = np.lexsort((targets, subject_indexes))
    inputs, targets, subject_indexes = inputs[sorted_indices], targets[sorted_indices], subject_indexes[sorted_indices]

    # Initialize folds
    folds = [[] for _ in range(num_folds)]
    
    # Distribute samples across folds
    for i, (x, y, s) in enumerate(zip(inputs, targets, subject_indexes)):
        folds[i % num_folds].append((x, y, s))  # Cycle through folds
    
    # Convert to NumPy arrays
    final_folds = [(np.array([x for x, _, _ in fold]), 
                    np.array([y for _, y, _ in fold]), 
                    np.array([s for _, _, s in fold])) for fold in folds]
    
    print(f"Fold sizes: {[len(fold[1]) for fold in final_folds]}")

    return final_folds

def create_objective(folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device: str, max_epochs: int = 1000, patience_start: int = 20, save_basepath: str = None, interval: int = 10) -> callable:
    """
    Create an objective function for optuna to optimise hyperparameters
    
    :param folds: The folds to use for cross-validation
    :param device: The device to use
    :param max_epochs: The maximum number of epochs
    :param patience_start: The number of epochs to wait before early stopping
    :param save_basepath: The base path to save the models and training losses (Deprecated for memory reasons)
    :param interval: The interval to output the training and validation loss
    :return: The objective function
    """
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        #temporal_kernel = trial.suggest_int('temporal_kernel', 5, 19, step=2) # Odd number
        temporal_kernel = trial.suggest_int('temporal_kernel', 9, 19, step=2)
        num_kernel = trial.suggest_categorical('num_kernel', [32, 64, 128])
        depth = trial.suggest_int('depth', 2, 8, step=2)
        heads = trial.suggest_int('heads', 8, 16, step=4)
        mlp_dim = trial.suggest_int('mlp_dim', 16, 64, step=8)
        dim_head = trial.suggest_int('dim_head', 16, 64, step=8)

        print(f"Starting trial with\nlr: {lr}, batch_size: {batch_size}, dropout: {dropout}, temporal_kernel: {temporal_kernel}, num_kernel: {num_kernel}, depth: {depth}, heads: {heads}, mlp_dim: {mlp_dim}, dim_head: {dim_head}")

        num_classes = len(np.unique(folds[0][1]))
        
        fold_losses = []
        fold_sizes = [] # Used to take a weighted average of the losses
        fold_accs = []
        fold_f1s = []


        for i, (test_inputs, test_targets, _) in enumerate(folds):
            print(f"Fold {i+1}/{len(folds)}")

            # Use the i'th fold as the validation set, and the rest as the training set
            fold_idxs = [j for j in range(len(folds)) if j != i]
            train_inputs = np.concatenate([folds[j][0] for j in fold_idxs])
            train_targets = np.concatenate([folds[j][1] for j in fold_idxs])
            train_subjects = np.concatenate([folds[j][2] for j in fold_idxs])

            train_inputs, train_targets, train_subjects, val_inputs, val_targets, val_subjects = split_data(train_inputs, train_targets, train_subjects, split=0.8)

            print(f"Training set size: {len(train_inputs)}, Validation set size: {len(val_inputs)}, Test set size: {len(test_inputs)}")

            



            model = EEGDeformer.Deformer(
                num_chan=14,
                num_time=FREQ*TIME,
                temporal_kernel=temporal_kernel,
                num_kernel=num_kernel,
                num_classes=num_classes,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                dropout=dropout
            )

            model.to(device)
            train_loader, val_loader = get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=batch_size)

            _, val_losses, val_accs, val_f1s = train_model(model, device, train_loader, val_loader, lr=lr, lr_dropoff_factor=0.65, max_epochs=max_epochs, patience_start=patience_start, save_path=None, interval=interval)

            print(f"Fold {i+1}/{len(folds)}: Validation loss: {min(val_losses)}")
            fold_losses.append(min(val_losses))
            fold_sizes.append(len(val_targets))

            dev_loader = DataLoader(EEGDataset(test_inputs, test_targets), batch_size=64, shuffle=False) # Batch size doesn't matter here, 64 seems to run faster
            model.eval()
            with torch.no_grad():
                batch_sizes = []
                batch_accs = []
                for batch in val_loader:
                    outputs = model(batch['inputs'].to(device))
                    # Calculate accuracy
                    fold_acc = (torch.argmax(outputs, dim=1) == batch['targets'].to(device)).sum().item() / len(batch['targets'])
                    batch_sizes.append(len(batch['targets']))
                    batch_accs.append(fold_acc)

            fold_accs.append(np.average(batch_accs, weights=batch_sizes))

            print(f"Fold {i+1}/{len(folds)}: accuracy on dev set: {fold_accs[-1]}")

            """# Save training and val losses in jsons
            with open(os.path.join(save_basepath, f"train_losses_{trial.number}_{i}.json"), 'w') as f:
                json.dump(train_losses, f)

            with open(os.path.join(save_basepath, f"val_losses_{trial.number}_{i}.json"), 'w') as f:
                json.dump(val_losses, f)"""
            
        print(f"Average validation loss: {np.average(fold_losses, weights=fold_sizes)}")
        print(f"Average accuracy: {np.average(fold_accs, weights=fold_sizes)}")
        return np.average(fold_accs, weights=fold_sizes)
    
    return objective

def plot_accuracy(class_results: dict[str, dict[str, int]], sort: bool = False) -> None:
    """
    Draw a bar chart of the class accuracies
    Draw a line at the overall accuracy

    :param accuracy: The overall accuracy
    :param class_accuracies: The accuracy per class
    """
    # Calculate accuracy from class results
    total_correct = 0
    total = 0
    class_accuracies = {}
    for label in class_results:
        correct = class_results[label]['correct']
        total_samples = class_results[label]['total']
        total_correct += correct
        total += total_samples
        class_accuracies[label] = correct / total_samples
    
    if sort:
        class_accuracies = {k: v for k, v in sorted(class_accuracies.items(), key=lambda item: item[1])}

    accuracy = total_correct / total
    print(f"Posteriori accuracy: {accuracy}")
    plt.bar(class_accuracies.keys(), class_accuracies.values(), zorder=3)
    plt.axhline(y=accuracy, color='r', linestyle='--', label='Overall accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(visible=True, which='major', axis='y', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.title('Accuracy per class')
    plt.show()

def plot_subject_accuracy(results: dict[str, list]):
    """
    Draw a bar chart of the accuracy per subject
    Draw a line at the overall accuracy

    :param accuracy: The overall accuracy
    :param class_accuracies: The accuracy per class
    :param subject_accuracies: The accuracy per subject
    """

    subjects = np.unique(results['subject'])
    print("Subjects:", subjects)
    subject_accuracies = {subject: {'correct': 0, 'total': 0} for subject in subjects}
    for i in range(len(results['true'])):
        # Change from tensor to int
        subject = results['subject'][i].item()
        subject_accuracies[subject]['total'] += 1
        if results['true'][i] == results['pred'][i]:
            subject_accuracies[subject]['correct'] += 1

    for subject in subject_accuracies:
        print(f"Subject {subject}, correct: {subject_accuracies[subject]['correct']}, total: {subject_accuracies[subject]['total']}")
        subject_accuracies[subject] = subject_accuracies[subject]['correct'] / subject_accuracies[subject]['total']

    plt.bar(subject_accuracies.keys(), subject_accuracies.values(), zorder=3)
    plt.axhline(y=np.mean(list(subject_accuracies.values())), color='r', linestyle='--', label='Overall accuracy')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.xticks(ticks=subjects, rotation=0)
    plt.legend()
    plt.grid(visible=True, which='major', axis='y', linestyle='-', zorder=0)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.title('Accuracy per subject')
    plt.show()

    

def plot_loss(val_losses: list[float], train_losses: list[float]):
    # Create 10 epoch moving average to display alongside raw data
    val_moving_avg = []
    for i in range(len(val_losses)):
        if i < 10:
            # Use average of all previous epochs
            val_moving_avg.append(sum(val_losses[:i+1]) / (i+1))
        else:
            val_moving_avg.append(sum(val_losses[i-10:i]) / 10)



    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_moving_avg, label='Validation Loss (10 epoch moving average)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(visible=True, which='major', axis='y', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


def channel_activation(channel_names: list[str], data: dict[str, np.ndarray], subject: Union[int, None], class_label: Union[int, None]) -> None:
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

    info = mne.create_info(ch_names=channel_names, sfreq=FREQ, ch_types='eeg')
    evoked = mne.EvokedArray(mean_data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

    evoked.plot_topomap(times=0, ch_type='eeg', show_names=True, outlines='head', show=False, cmap = 'viridis')
    plt.show()

def split_unseen_subjects(inputs: np.ndarray, targets: np.ndarray, idx: np.ndarray, unseen_subjects: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and test sets, ensuring that the subjects in the test set are not present in the training set

    :param inputs: The inputs
    :param targets: The targets
    :param idx: The subject indexes
    :param unseen_subjects: The subjects to exclude from the training set
    :return: The training inputs, training targets, training subject indexes, test inputs, test targets, test subject indexes
    """
    train_inputs = []
    train_targets = []
    train_idx = []

    test_inputs = []
    test_targets = []
    test_idx = []

    for input, target, subject in zip(inputs, targets, idx):
        if subject in unseen_subjects:
            test_inputs.append(input)
            test_targets.append(target)
            test_idx.append(subject)
        else:
            train_inputs.append(input)
            train_targets.append(target)
            train_idx.append(subject)

    return train_inputs, train_targets, train_idx, np.array(test_inputs), np.array(test_targets), np.array(test_idx)

def subject_folds(idx:np.ndarray, num_folds:int = 5, shuffle: bool = True) -> list[list[int]]:
    """
    Create folds based on the subjects

    :param idx: The subject indexes
    :param num_folds: The number of folds to create
    :param shuffle: Whether to shuffle the subjects before creating the folds
    :return: A list of lists containing the subject indexes for each fold
    """
    unique_subjects = np.unique(idx)
    
    if shuffle:
        np.random.shuffle(unique_subjects)

    folds = [[] for _ in range(num_folds)]
    for i, subject in enumerate(unique_subjects):
        folds[i % num_folds].append(subject)

    print(f"Fold sizes: {[len(fold) for fold in folds]}")
    print(f"Folds: {folds}")

    return folds

def fold_distribution(folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], class2lab: dict[str, int]) -> None:
    """
    Plot the distribution of the classes in the folds

    :param folds: The folds to use
    :param class2lab: The mapping from class to label
    """

    fold_distributions = []
    for fold in folds:
        dist = {}
        for _, target, subject in zip(fold[0], fold[1], fold[2]):
            target = list(class2lab.keys())[list(class2lab.values()).index(target)]
            try:
                dist[f"{subject}_{target}"] += 1
            except KeyError:
                dist[f"{subject}_{target}"] = 1

        fold_distributions.append(dist)
            

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, fold in enumerate(fold_distributions):
        ax = axs[i // 3, i % 3]
        ax.bar(fold.keys(), fold.values())
        ax.set_title(f"Fold {i+1}")

    # Disable unused axes
    for i in range(len(folds), 9):
        axs[i // 3, i % 3].axis('off')

    plt.show()

class InstanceNormEEG(torch.nn.Module):
    def __init__(self, num_channels):
        super(InstanceNormEEG, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm1d(num_channels, affine=True)  # Normalize each EEG channel separately

    def forward(self, x):
        return self.instance_norm(x)
    

def kfold_train(test_subfolder: str, inputs: np.ndarray, targets: np.ndarray, idx: np.ndarray, class2lab: dict[str, int], params: dict[str, Union[int, float]], num_folds: int, validation_split: float, max_epochs: int, device: str, unseen_testing: bool = False, base_model_name: str = None) -> None:
    """
    Train the model using k-fold cross-validation
    
    :param test_subfolder: The subfolder to save the results in
    :param inputs: The inputs to use for training
    :param targets: The targets to use for training
    :param idx: The subject indexes to use for training
    :param class2lab: The mapping from class to label
    :param params: The parameters to use for training
    :param num_folds: The number of folds to use for cross-validation
    :param validation_split: The fraction of the training data to use for validation
    :param max_epochs: The maximum number of epochs to train for
    :param device: The device to use for training
    :param unseen_testing: Whether to use unseen testing or not
    :param base_model_name: The name of the base model to use for transfer learning, if None, a new model will be created
    
    """
    test_path = os.path.join(os.path.dirname(__file__), "test")
    os.makedirs(test_path, exist_ok=True)
    """test_data_path = os.path.join(test_path, "test_data")
    os.makedirs(test_data_path, exist_ok=True)"""

    test_subfolder = os.path.join(test_path, test_subfolder)
    data_subfolder = os.path.join(test_subfolder, "data")
    os.makedirs(data_subfolder, exist_ok=True)
    os.makedirs(test_subfolder, exist_ok=True)


    #save_base = "baseline_all_subjects"
    save_base = "deformer"
    
    #folds = create_folds(inputs, targets, idx, num_folds=5) # Test folds, 5 folds gives an 80/20 split
    #folds = subject_folds(idx, num_folds=5, shuffle=True)

    if base_model_name is not None:
        transfer_learning = True
    else:
        transfer_learning = False

    if unseen_testing:
        folds = subject_folds(idx, num_folds=num_folds, shuffle=False)       # 4/5 subjects per 'fold' (16/17 training, 5/4 testing)
    else:
        # TODO adapt
        #inputs = InstanceNormEEG(14)(torch.tensor(inputs, dtype=torch.float32)).detach().cpu().numpy() # Instance norm didn't work :c gave worse results, no time to try other methods
        folds = create_folds(inputs, targets, idx, num_folds=num_folds) # Test folds, 5 folds gives an 80/20 split
    

    best_performer = {"Accuracy": 0, "Fold": 0}

    # Change depending on native task or transfer learning source
    if transfer_learning:
        num_classes = 4
    else:
        num_classes = len(class2lab)

    #fold_distribution(folds, class2lab)

    for i, fold in enumerate(folds):

        # Create new model
        model = EEGDeformer.Deformer(
            num_chan=14,
            num_time=FREQ*TIME,
            temporal_kernel=params['temporal_kernel'],
            num_kernel=params['num_kernel'],
            num_classes= num_classes,
            depth=params['depth'],
            heads=params['heads'],
            mlp_dim=params['mlp_dim'],
            dim_head=params['dim_head'],
            dropout=params['dropout']
        )

        # Load VI base
        if transfer_learning:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), base_model_name)))
            model.change_head(20)

        model.to(device)        # To GPU

        print(f"Using fold {i+1}/{len(folds)} as test set")

        if unseen_testing:
            print(f"Unseen participants: {fold}")
            train_inputs, train_targets, train_idx, test_inputs, test_targets, test_idx = split_unseen_subjects(inputs, targets, idx, fold)
        else:
            fold_idx = [j for j in range(len(folds)) if j != i] # Get indexes of all folds except the current fold
            train_inputs = np.concatenate([folds[j][0] for j in fold_idx])      # Inputs
            train_targets = np.concatenate([folds[j][1] for j in fold_idx])     # Targets
            train_idx = np.concatenate([folds[j][2] for j in fold_idx])         # Subject indexes (for stratification)
            
            test_inputs = folds[i][0]       # Use current fold as test set
            test_targets = folds[i][1]
            test_idx = folds[i][2]


        # Split training data into training and validation sets
        train_inputs, train_targets, train_idx, val_inputs, val_targets, val_idx = split_data(train_inputs, train_targets, train_idx, split=validation_split, shuffle=True) # 15% validation set to maximise training data
        train_loader, val_loader = get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=params['batch_size'])

        print(f"Training samples: {len(train_targets)}, Validation samples: {len(val_targets)}, Test samples: {len(test_targets)}")
        save_path = os.path.join(test_subfolder, f"{save_base}_fold_{i}.pth")
                                
        train_dataset = EEGDataset_subjectID(train_inputs, train_targets, train_idx)
        val_dataset = EEGDataset_subjectID(val_inputs, val_targets, val_idx)
        torch.save(train_dataset, os.path.join(data_subfolder, f"{save_base}_fold_{i}_train_dataset.pth"))
        torch.save(val_dataset, os.path.join(data_subfolder, f"{save_base}_fold_{i}_val_dataset.pth"))
        del train_dataset, val_dataset

        test_dataset = EEGDataset_subjectID(test_inputs, test_targets, test_idx)
        torch.save(test_dataset, os.path.join(data_subfolder, f"{save_base}_fold_{i}_dataset.pth"))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        if transfer_learning:
            head_lr = params['lr']
            base_lr = head_lr / 8
            train_losses, val_losses, val_accs, val_f1 = fine_tune(model, device, train_loader, val_loader, base_lr=base_lr, head_lr=head_lr, max_epochs=max_epochs, patience_start=25, save_path=save_path, interval=5)
        else:
            train_losses, val_losses, val_accs, val_f1 = train_model(model, device, train_loader, val_loader, lr=params['lr'], lr_dropoff_factor=0.65, max_epochs=max_epochs, patience_start=25, label_smoothing=0.1, save_path=save_path, interval=5)
        # Save training and val loss in json
        json_loss = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'val_f1': val_f1,
            'parameters': params
        }

        if unseen_testing:
            json_loss['unseen_subjects'] = [int(subject) for subject in fold]

        with open(os.path.join(test_subfolder, f"{save_base}_fold_{i}_loss.json"), 'w') as f:
            json.dump(json_loss, f)

        # Evaluate the model
        
        results = evaluate(model, test_loader, device)

        with open(os.path.join(test_subfolder, f"{save_base}_fold_{i}_results.json"), 'w') as f:
            json.dump(results, f)

        results['pred'] = np.array(results['pred'])
        results['true'] = np.array(results['true'])

        accuracy = (results['pred'] == results['true']).sum() / len(results['true'])

        if accuracy > best_performer["Accuracy"]:
            best_performer["Accuracy"] = accuracy
            best_performer["Fold"] = i+1

        #json_results = {
        #    'accuracy': accuracy,
        #    'class_accuracies': class_accuracies,
        #    'results': results
        #} # Just test later I can't be bothered to fix this, use the corresponding inputs targets and subject ids saved below    

        #with open(os.path.join(test_path, f"{save_base}_fold_{i}_results.json"), 'w') as f:
        #    json.dump(json_results, f)

        print(f"Fold {i+1}/{len(folds)}: Accuracy: {accuracy}")

        # Save model
        torch.save(model.state_dict(), save_path)
        # Save test data (just in case)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_inputs.npy"), test_inputs)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_targets.npy"), test_targets)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_idx.npy"), test_idx)
        
    
    # Save params
    with open(os.path.join(test_subfolder, f"{save_base}_hyperparams.json"), 'w') as f:
        json.dump(params, f)


    print(f"Best performer: Fold {best_performer['Fold']} with accuracy {best_performer['Accuracy']}")

def unseen_subject_kfold(test_subfolder: str, inputs: np.ndarray, targets: np.ndarray, idx: np.ndarray, class2lab: dict[str, int], params: dict[str, Union[int, float]], num_folds: int, max_epochs: int, device: str, unseen_subj: int, base_model_name: str = None) -> None:
    """
    Train the model using k-fold cross-validation, using a specific subject as the test set
    
    :param test_subfolder: The subfolder to save the results in
    :param inputs: The inputs to use for training
    :param targets: The targets to use for training
    :param idx: The subject indexes to use for training
    :param class2lab: The mapping from class to label
    :param params: The parameters to use for training
    :param num_folds: The number of folds to use for cross-validation
    :param max_epochs: The maximum number of epochs to train for
    :param device: The device to use for training
    :param unseen_subj: The subject to use for unseen testing
    :param base_model_name: The name of the base model to use for transfer learning, if None, a new model will be created
    
    """
    test_path = os.path.join(os.path.dirname(__file__), "test")
    os.makedirs(test_path, exist_ok=True)
    """test_data_path = os.path.join(test_path, "test_data")
    os.makedirs(test_data_path, exist_ok=True)"""

    test_subfolder = os.path.join(test_path, test_subfolder)
    data_subfolder = os.path.join(test_subfolder, "data")
    os.makedirs(data_subfolder, exist_ok=True)
    os.makedirs(test_subfolder, exist_ok=True)


    #save_base = "baseline_all_subjects"
    save_base = "deformer"
    
    #folds = create_folds(inputs, targets, idx, num_folds=5) # Test folds, 5 folds gives an 80/20 split
    #folds = subject_folds(idx, num_folds=5, shuffle=True)

    if base_model_name is not None:
        transfer_learning = True
    else:
        transfer_learning = False

    min_subj_idx = np.min(idx)
    max_subj_idx = np.max(idx)

    if unseen_subj < min_subj_idx or unseen_subj > max_subj_idx:
        raise ValueError(f"Unseen subject {unseen_subj} is not in the range of subjects {min_subj_idx}-{max_subj_idx}")

    # Separate unseen subject from the rest of the data
    seen_inputs, seen_targets, seen_idx, unseen_inputs, unseen_targets, unseen_idx = split_unseen_subjects(inputs, targets, idx, [unseen_subj])

    # Create folds based on the subjects
    folds = create_folds(seen_inputs, seen_targets, seen_idx, num_folds=num_folds) # Test folds, 5 folds gives an 80/20 split
    #folds = subject_folds(seen_idx, num_folds=num_folds, shuffle=True)       # 4/5 subjects per 'fold' (16/17 training, 5/4 testing)
    

    best_performer = {"Accuracy": 0, "Fold": 0}

    # Change depending on native task or transfer learning source
    if transfer_learning:
        num_classes = 4
    else:
        num_classes = len(class2lab)

    #fold_distribution(folds, class2lab)

    for i, fold in enumerate(folds):

        # Create new model
        model = EEGDeformer.Deformer(
            num_chan=14,
            num_time=FREQ*TIME,
            temporal_kernel=params['temporal_kernel'],
            num_kernel=params['num_kernel'],
            num_classes= num_classes,
            depth=params['depth'],
            heads=params['heads'],
            mlp_dim=params['mlp_dim'],
            dim_head=params['dim_head'],
            dropout=params['dropout']
        )

        # Load VI base
        if transfer_learning:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), base_model_name)))
            model.change_head(20)

        model.to(device)        # To GPU

        print(f"Using fold {i+1}/{len(folds)} as test set")

        #train_inputs, train_targets, train_idx, val_inputs, val_targets, val_idx = split_unseen_subjects(seen_inputs, seen_targets, seen_idx, fold)
        #train_inputs, train_targets, train_idx = np.array(train_inputs), np.array(train_targets), np.array(train_idx)
        fold_idx = [j for j in range(len(folds)) if j != i] # Get indexes of all folds except the current fold
        train_inputs = np.concatenate([folds[j][0] for j in fold_idx])      # Inputs
        train_targets = np.concatenate([folds[j][1] for j in fold_idx])     # Targets
        train_idx = np.concatenate([folds[j][2] for j in fold_idx])         # Subject indexes (for stratification)
            
        val_inputs = folds[i][0]       # Use current fold as validation
        val_targets = folds[i][1]
        val_idx = folds[i][2]


        # Split training data into training and validation sets
        train_loader, val_loader = get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=params['batch_size'])
        print(f"Training samples: {len(train_targets)}, Validation samples: {len(val_targets)}, Test samples: {len(unseen_targets)}")

        save_path = os.path.join(test_subfolder, f"{save_base}_fold_{i}.pth")

        test_dataset = EEGDataset_subjectID(unseen_inputs, unseen_targets, unseen_idx)
        torch.save(test_dataset, os.path.join(data_subfolder, f"{save_base}_fold_{i}_dataset.pth"))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        if transfer_learning:
            head_lr = params['lr']
            base_lr = head_lr / 8
            train_losses, val_losses, val_accs, val_f1 = fine_tune(model, device, train_loader, val_loader, base_lr=base_lr, head_lr=head_lr, max_epochs=max_epochs, patience_start=25, save_path=save_path, interval=5)
        else:
            train_losses, val_losses, val_accs, val_f1 = train_model(model, device, train_loader, val_loader, lr=params['lr'], lr_dropoff_factor=0.65, max_epochs=max_epochs, patience_start=25, label_smoothing=0.1, save_path=save_path, interval=5)
        # Save training and val loss in json
        json_loss = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'val_f1': val_f1,
            'parameters': params
        }

        json_loss['unseen_subjects'] = [unseen_subj]

        with open(os.path.join(test_subfolder, f"{save_base}_fold_{i}_loss.json"), 'w') as f:
            json.dump(json_loss, f)

        # Evaluate the model
        
        results = evaluate(model, test_loader, device)

        with open(os.path.join(test_subfolder, f"{save_base}_fold_{i}_results.json"), 'w') as f:
            json.dump(results, f)

        results['pred'] = np.array(results['pred'])
        results['true'] = np.array(results['true'])

        accuracy = (results['pred'] == results['true']).sum() / len(results['true'])

        if accuracy > best_performer["Accuracy"]:
            best_performer["Accuracy"] = accuracy
            best_performer["Fold"] = i+1

        #json_results = {
        #    'accuracy': accuracy,
        #    'class_accuracies': class_accuracies,
        #    'results': results
        #} # Just test later I can't be bothered to fix this, use the corresponding inputs targets and subject ids saved below    

        #with open(os.path.join(test_path, f"{save_base}_fold_{i}_results.json"), 'w') as f:
        #    json.dump(json_results, f)

        print(f"Fold {i+1}/{len(folds)}: Accuracy: {accuracy}")

        # Save model
        torch.save(model.state_dict(), save_path)
        # Save test data (just in case)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_inputs.npy"), test_inputs)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_targets.npy"), test_targets)
        #np.save(os.path.join(data_subfolder, f"{save_base}_fold_{i}_idx.npy"), test_idx)
        
    
    # Save params
    with open(os.path.join(test_subfolder, f"{save_base}_hyperparams.json"), 'w') as f:
        json.dump(params, f)


    print(f"Best performer: Fold {best_performer['Fold']} with accuracy {best_performer['Accuracy']}")


    

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

    print(f"Number of classes: {len(class2lab)}")

    ignore_subject = None

    inputs, targets, idx, ch_names = load_data(class2lab, ignore_class=None, ignore_subject=ignore_subject)

    

    #inputs, targets, idx, unseen_in, unseen_targets, unseen_idx = split_unseen_subjects(inputs, targets, idx, [18,19,20])
    #train_inputs, train_targets, train_idx, test_inputs, test_targets, test_idx = split_data(inputs, targets, idx, split=0.75, shuffle=True)

    """print(f"Number of training samples: {len(train_targets)}")
    print(f"Number of test samples: {len(test_targets)}")

    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets: {train_targets}")"""

    #folds = create_folds(train_inputs, train_targets, train_idx, num_folds=5) # 7 Folds should give 2 samples per class per subject in each fold (+ 15 training)
    #print(f"Number of folds: {len(folds)}")
    #print(f"Number of samples in each fold: {[len(fold[1]) for fold in folds]}")


    save_basepath = os.path.join(os.path.dirname(__file__), "optuna_models")
    os.makedirs(save_basepath, exist_ok=True)

    #objective = create_objective(train_inputs, train_targets, val_inputs, val_targets, device, max_epochs=1000, patience_start=20, save_basepath=save_basepath)
    #objective = create_objective(folds, device, max_epochs=1000, patience_start=40, save_basepath=save_basepath, interval=50)

    study_db_path = os.path.join(os.path.dirname(__file__), "deformer_study.db")


    study = optuna.create_study(direction='maximize', study_name='quick_tune', storage='sqlite:///{}'.format(study_db_path), load_if_exists=True)

    

    best_params = study.best_params




    print(f"\nUsing best parameters:\n{best_params}\n")
    """kfold_train(
        test_subfolder="VI unseen LOSO",
        inputs=inputs,
        targets=targets,
        idx=idx,
        class2lab=class2lab,
        params={'lr': 0.005, 'batch_size': 32, 'dropout': 0.25, 'temporal_kernel': 11, 'num_kernel': 128, 'depth': 6, 'heads': 12, 'mlp_dim': 24, 'dim_head': 64},
        num_folds=21,
        validation_split=0.8, #0.85 for seen testing
        max_epochs=1000,
        device=device,
        unseen_testing=True,
        base_model_name="deformer_base_VI.pth"
    )"""

    
    unseen_subject_kfold(
        test_subfolder="VI subj 4 unseen",
        inputs=inputs,
        targets=targets,
        idx=idx,
        class2lab=class2lab,
        params={'lr': 0.005, 'batch_size': 32, 'dropout': 0.25, 'temporal_kernel': 11, 'num_kernel': 128, 'depth': 6, 'heads': 12, 'mlp_dim': 24, 'dim_head': 64},
        num_folds=5,
        max_epochs=1000,
        device=device,
        unseen_subj=4,
        base_model_name="deformer_base_VI.pth"
    )


if __name__ == "__main__":
    main()



