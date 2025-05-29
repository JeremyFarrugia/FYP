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
    data_dir = os.path.join(sc_dir, "standardised") # Folder contraining the standardised data

    inputs = []
    targets = []
    subject_indexes = []

    channel_names = [] # Channel names for the topomap

    tmin = 0 # Start at the onset of the trigger
    tmax = 40 # Take 40s of data after the trigger
    # TODO: might have to change how epoch windows are segmented to be aware of removals due to bad segments

    c = 0
    for file in tqdm(os.listdir(data_dir), desc='Loading data'):
        # Load .fif file
        raw = mne.io.read_raw_fif(os.path.join(data_dir, file), preload=True, verbose=False)

        channel_names = raw.ch_names # This gets updated each time but doesn't matter as all channel names should be the same

        # Segment the data by taking 4s of data after each trigger (from annotations)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        #epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=4, baseline=None, preload=True, verbose=False) # NOTE: I change tmax to 4s from 40s as I changed the way the data is segmented

        subject_index = int(file.split('_')[0][3:])

        if ignore_subject is not None and subject_index in ignore_subject:
            continue

        for event_label, _ in event_id.items():
            event_epochs = epochs[event_label]

            target_label: int = class2lab.get(event_label, -1)

            if target_label == -1:
                continue

            if ignore_class is not None and target_label in ignore_class:
                continue

            
            for i, epoch in enumerate(event_epochs):
                """windows = 
                inputs.extend(windows)
                targets.extend([target_label] * len(windows))
                subject_indexes.extend([subject_index] * len(windows))"""
                inputs.append(epoch)
                targets.append(target_label)
                subject_indexes.append(subject_index)
        c+=1
        if end is not None and c >= end:
            break

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_dropoff_factor, patience=15, verbose=True) # Reduce learning rate on plateau
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6) # Cosine annealing with restarts
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing) # Label smoothing to prevent overfitting

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
        val_accs.append(np.average(val_acc))
        val_f1s.append(np.average(val_f1))

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

    return train_losses, val_losses # TODO return val_accs and val_f1s

def evaluate(model: torch.nn.Module, labels: list[str], data_loader: DataLoader, device: str) -> tuple[float, dict[str, float]]:
    """
    Evaluate the model on the validation set

    :param model: The model to evaluate
    :param val_loader: The validation data loader
    :return: The accuracy of the model on the validation set and the accuracy per class
    """
    model.eval()
    correct_per_class = {label: 0 for label in labels}
    total_per_class = {label: 0 for label in labels}
    correct = 0
    total = 0
    results = {'pred': [], 'true': [], 'subject': []}
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
                results['pred'].append(labels[predicted[i]])
                results['true'].append(labels[batch['targets'][i]])
                results['subject'].append(batch['subjects'][i])
    
    return correct / total, {label: {'correct': correct_per_class[label], 'total': total_per_class[label]} for label in labels if total_per_class[label] > 0}, results

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

def create_folds(inputs: list[np.ndarray], targets: list[int], subject_indexes: list[int], num_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create stratified folds for cross-validation

    :param inputs: The inputs
    :param targets: The targets
    :param subject_indexes: The subject indexes
    :param num_folds: The number of folds
    :return: A list of tuples containing the data (inputs, targets, subject indexes) for each fold
    """

    stratified_data = stratify_data(inputs, targets, subject_indexes)

    folds = []
    for i in range(num_folds+1):
        fold_inputs = []
        fold_targets = []
        fold_subject_indexes = []
        if i < num_folds:
            print(f"Creating fold {i+1}/{num_folds}")
        else:
            print(f"Creating temporary fold to redistribute data")

        for subject in stratified_data:
            for label in stratified_data[subject]:
                data = stratified_data[subject][label]
                fold_size = len(data) // num_folds
                start = i * fold_size
                #end = (i+1) * fold_size if i < num_folds - 1 else len(data)
                end = (i+1) * fold_size
                if i < num_folds:
                    fold_data = data[start:end]
                else:
                    start = i * fold_size # Start where the last fold ended
                    fold_data = data[start:]

                fold_inputs.extend(fold_data)
                fold_targets.extend([label] * len(fold_data))
                fold_subject_indexes.extend([subject] * len(fold_data))
        if i < num_folds:
            folds.append((fold_inputs, fold_targets, fold_subject_indexes))
        else: # 'Extra temporary fold' to redistribute the data
            # Randomly distribute the additional data
            print(f"Redistributing data, amount of data: {len(fold_inputs)}")
            # Redistribute the data to the other folds
            for data_idx in range(len(fold_inputs)):
                target_fold = random.randint(0, num_folds-1)
                target_fold_inputs, target_fold_targets, target_fold_subject_indexes = folds[target_fold]
                target_fold_inputs.append(fold_inputs[data_idx])
                target_fold_targets.append(fold_targets[data_idx])
                target_fold_subject_indexes.append(fold_subject_indexes[data_idx])
                folds[target_fold] = (target_fold_inputs, target_fold_targets, target_fold_subject_indexes)
    
    print(f"Fold sizes: {[len(fold[1]) for fold in folds]}")

    # Convert folds to numpy arrays
    folds = [(np.array(inputs), np.array(targets), np.array(subject_indexes)) for inputs, targets, subject_indexes in folds]
        
    return folds

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    class2lab = {
        "Apple":1,
        "Car":2,
        "Flower":3,
        "Face":4,
    }
    

    # Set class2lab keys to lowercase
    #class2lab = {key.lower(): value for key, value in class2lab.items()}
    # -1 from values to make them 0-indexed
    class2lab = {key: value-1 for key, value in class2lab.items()}

    print(f"Number of classes: {len(class2lab)}")


    inputs, targets, idx, ch_names = load_data(class2lab, ignore_class=None, ignore_subject=None) # https://data.mendeley.com/datasets/g9shp2gxhy/2
    train_inputs, train_targets, train_idx, test_inputs, test_targets, test_idx = split_data(inputs, targets, idx, split=0.8, shuffle=True)

    print(f"Number of training samples: {len(train_targets)}")
    print(f"Number of test samples: {len(test_targets)}")

    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets: {train_targets}")

    #folds = create_folds(train_inputs, train_targets, train_idx, num_folds=7) # 7 Folds should give 2 samples per class per subject in each fold (+ 15 training)
    #print(f"Number of folds: {len(folds)}")
    #print(f"Number of samples in each fold: {[len(fold[1]) for fold in folds]}")


    save_basepath = os.path.join(os.path.dirname(__file__), "optuna_models")
    os.makedirs(save_basepath, exist_ok=True)


    params = {'lr': 0.005, 'batch_size': 32, 'dropout': 0.25, 'temporal_kernel': 11, 'num_kernel': 128, 'depth': 6, 'heads': 12, 'mlp_dim': 24, 'dim_head': 64}



    # Best parameters: {'lr': 0.007765876411213384, 'batch_size': 64, 'dropout': 0.40015155108112843, 'temporal_kernel': 15, 'num_kernel': 64, 'depth': 5, 'heads': 12, 'mlp_dim': 54, 'dim_head': 34}

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



    #load_base_model(model, os.path.join(os.path.dirname(__file__), "deformer_base.pth"), len(class2lab))



    model.to(device)

    val_inputs, val_targets = test_inputs, test_targets
    train_loader, val_loader = get_loaders(train_inputs, train_targets, val_inputs, val_targets, batch_size=params['batch_size'])

    save_path = os.path.join(os.path.dirname(__file__), "deformer_base_VI_tuned.pth")
    train_losses, val_losses = train_model(model, device, train_loader, val_loader, lr=params['lr'], lr_dropoff_factor=0.65, max_epochs=1000, patience_start=50, label_smoothing=0.1, save_path=save_path)
    #train_losses, val_losses = fine_tune(model, device, train_loader, val_loader, base_lr=1e-5, head_lr=1e-3, max_epochs=1000, patience_start=50, save_path=os.path.join(os.path.dirname(__file__), "deformer_finetuned.pth"))
    
    test_loader = DataLoader(EEGDataset_subjectID(test_inputs, test_targets, test_idx), batch_size=64, shuffle=False)
    accuracy, class_accuracies, results = evaluate(model, list(class2lab.keys()), test_loader, device)

    print(f"Accuracy: {accuracy}")
    print(f"Class accuracies:\n{class_accuracies}")

    plot_accuracy(class_accuracies)
    plot_subject_accuracy(results)

    plot_loss(val_losses, train_losses)

    # Confusion matrix from results
    cm = confusion_matrix(results['true'], results['pred'], labels=list(class2lab.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class2lab.keys()))
    disp.plot()
    plt.show()

    print(f"Total parameters: {EEGDeformer.count_parameters(model)}")

if __name__ == "__main__":
    main()
