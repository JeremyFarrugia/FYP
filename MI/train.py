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
import time
from sklearn.metrics import f1_score


SEED = 42
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
    parent_dir = os.path.dirname(sc_dir)
    #data_dir = os.path.join(parent_dir, "segmented_data") # Folder contraining the standardised segmented data
    data_dir = os.path.join(parent_dir, "standardised_data") # Folder contraining the standardised segmented data

    inputs = []
    targets = []
    subject_indexes = []

    tmin = 0 # Start at the onset of the trigger
    tmax = 4 # Take 4s of data after the trigger

    c = 0
    for file in tqdm(os.listdir(data_dir), desc='Loading data'):
        subject_index = int(file.split('S')[1][:3]) # Get the subject index from the filename

        if ignore_subject is not None and subject_index in ignore_subject:
            continue

        # Load .fif file
        raw = mne.io.read_raw_fif(os.path.join(data_dir, file), preload=True, verbose=False)

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

    return inputs, targets, subject_indexes

def split_data(inputs: list[np.ndarray], targets: list[int], subject_indexes: list[int], split: float = 0.8, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the inputs and targets into training and validation sets (Train inputs, Train targets, Val inputs, Val targets)
    Ensures proportional representation of each class for each subject in both training and validation sets

    :param inputs: The inputs
    :param targets: The target labels for each input
    :param subject_indexes: The subject index for each input (Who the data was recorded from, used to ensure that subjects appear in both training and validation sets)
    :param split: The proportion of the data to use for training

    :return: The training inputs, the training targets, the validation inputs, the validation targets
    """
    if shuffle:
        indices = np.arange(len(targets))
        random.seed(SEED)
        np.random.seed(SEED)
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
    val_inputs = []
    val_targets = []

    # Allocate data according to quotas
    for i in range(len(inputs)):
        if quotas[subject_indexes[i]][targets[i]] > 0:
            train_inputs.append(inputs[i])
            train_targets.append(targets[i])
            quotas[subject_indexes[i]][targets[i]] -= 1
        else: # Once the training quota is satisfied, allocate the rest to the validation set
            val_inputs.append(inputs[i])
            val_targets.append(targets[i])

    

    return np.array(train_inputs), np.array(train_targets), np.array(val_inputs), np.array(val_targets)

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

def evaluate(model: torch.nn.Module, class2lab: dict, data_loader: DataLoader, device: str) -> tuple[float, dict[str, float]]:
    """
    Evaluate the model on the validation set

    :param model: The model to evaluate
    :param val_loader: The validation data loader
    :return: The accuracy of the model on the validation set and the accuracy per class
    """
    labels = list(class2lab.keys())
    model.eval()
    correct_per_class = {label: 0 for label in labels}
    total_per_class = {label: 0 for label in labels}
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['inputs'].to(device))
            _, predicted = torch.max(outputs, 1)
            total += batch['targets'].size(0)
            correct += (predicted == batch['targets'].to(device)).sum().item()

            for i in range(len(predicted)):
                label = labels[batch['targets'][i]]
                total_per_class[label] += 1
                if predicted[i] == batch['targets'][i]:
                    correct_per_class[label] += 1
    
    return correct / total, {label: correct_per_class[label] / total_per_class[label] for label in labels if total_per_class[label] > 0}

def create_objective(folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_folds: int, device: str, max_epochs: int = 1000, patience_start: int = 20, save_basepath: str = None, interval: int = 50) -> callable:
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
        dropout = trial.suggest_float('dropout', 0.3, 0.5, step=0.05)
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
        fold_accs = []
        fold_sizes = [] # Used to take a weighted average of the losses


        for i, (dev_inputs, dev_targets, _) in enumerate(folds):
            if i >= num_folds:
                break
            print(f"Fold {i+1}/{len(folds)}")

            # Use the i'th fold as the validation set, and the rest as the training set
            fold_idxs = [j for j in range(len(folds)) if j != i]
            train_inputs = np.concatenate([folds[j][0] for j in fold_idxs])
            train_targets = np.concatenate([folds[j][1] for j in fold_idxs])
            train_idxs = np.concatenate([folds[j][2] for j in fold_idxs])

            train_inputs, train_targets, val_inputs, val_targets = split_data(train_inputs, train_targets, train_idxs, split=0.8, shuffle=True)



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

            print(f"Fold {i+1}/{len(folds)}:\nLowest Validation loss: {min(val_losses)}, Highest Validation accuracy: {max(val_accs)}, Highest Validation F1: {max(val_f1s)}")

            # Test the model on the dev set
            dev_loader = DataLoader(EEGDataset(dev_inputs, dev_targets), batch_size=batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                dev_loss = []
                dev_acc = []
                dev_f1 = []
                dev_batch_sizes = []
                for batch in dev_loader:
                    outputs = model(batch['inputs'].to(device))
                    loss = torch.nn.CrossEntropyLoss()(outputs, batch['targets'].to(device))
                    loss = loss.item()
                    dev_acc.append((torch.argmax(outputs, dim=1) == batch['targets'].to(device)).sum().item() / len(batch['targets']))
                    # Calculate F1 score
                    dev_f1.append(f1_score(batch['targets'].cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted'))
                    dev_loss.append(loss)
                    dev_batch_sizes.append(len(batch['targets']))

                dev_loss = np.average(dev_loss, weights=dev_batch_sizes)
                dev_acc = np.average(dev_acc, weights=dev_batch_sizes)
                dev_f1 = np.average(dev_f1, weights=dev_batch_sizes)


            fold_accs.append(dev_acc)
            fold_sizes.append(len(val_targets))

            
        print(f"Average accuracy: {np.average(fold_accs, weights=fold_sizes)}")
        return np.average(fold_accs, weights=fold_sizes)
    
    return objective


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    class2lab = {
        'left_fist': 0,
        'right_fist': 1,
        'both_feet': 2,
        'both_fists': 3,
    } # Dataset: https://www.physionet.org/content/eegmmidb/1.0.0/#files-panel

    labels = ['rest', 'left_fist', 'right_fist', 'both_fists', 'both_feet']

    start = 20
    ignore_subject_idxs = [idx for idx in range(start, 120)] # Only consider first 10 subjects
    ignore_subject_idxs = None

    inputs, targets, idx = load_data(class2lab, end=None, ignore_class=None, ignore_subject=ignore_subject_idxs)
    print(f"Number of samples: {len(inputs)}")
    labels = ['left_fist', 'right_fist', 'both_fists', 'both_feet']
    train_inputs, train_targets, val_inputs, val_targets = split_data(inputs, targets, idx, split=0.8, shuffle=True)
    print(f"Number of training samples: {len(train_inputs)}, number of validation samples: {len(val_inputs)}")

    #print(f"Number of training samples: {len(train_targets)}")

    save_basepath = os.path.join(os.path.dirname(__file__), "optuna_models")


    study_db_path = os.path.join(os.path.dirname(__file__), "deformer_study.db")

    
    params = {'lr': 0.005, 'batch_size': 32, 'dropout': 0.25, 'temporal_kernel': 11, 'num_kernel': 128, 'depth': 6, 'heads': 12, 'mlp_dim': 24, 'dim_head': 64} #  # MI-VEP tuned params
    model = EEGDeformer.Deformer(
        num_chan=14, 
        num_time=FREQ*TIME,
        temporal_kernel=params['temporal_kernel'],
        num_kernel=params['num_kernel'],
        num_classes=len(class2lab),
        depth=params['depth'],
        heads=params['heads'],
        mlp_dim=params['mlp_dim'],
        dim_head=params['dim_head'],
        dropout=params['dropout']
    )
    model.to(device)

    train_loader, val_loader = get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=params['batch_size'])

    train_losses, val_losses, val_accs, val_f1s = train_model(model, device, train_loader, val_loader, lr=params['lr'], lr_dropoff_factor=0.65, max_epochs=1000, patience_start=25, save_path=os.path.join(os.path.dirname(__file__), "deformer_base_MI.pth"))

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "deformer_base_MI_backup.pth"))


    #print(f"Total parameters: {EEGDeformer.count_parameters(model)}")
    # Get class performance on the validation set
    accuracy, class_accuracies = evaluate(model, labels, val_loader, device)
    print(f"Accuracy: {accuracy}")
    print(f"Class accuracies:\n{class_accuracies}")






if __name__ == "__main__":
    main()







