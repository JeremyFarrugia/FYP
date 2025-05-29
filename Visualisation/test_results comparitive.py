import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, accuracy_score
from train import EEGDataset, EEGDataset_subjectID
import EEGDeformer
from scipy.stats import t

test_dir = os.path.join(os.path.dirname(__file__), 'test')

tests = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

#results = [f for f in os.listdir(os.path.join(test_dir, tests[0])) if os.path.isfile(os.path.join(test_dir, tests[0], f)) and f.endswith('.json')]
#print(f"Found {len(results)} results files in {os.path.join(test_dir, tests[0])}")

desired_tests = ["baseline all subjects tuned",
                 "MI all subjects",
                 "VI all subjects",
]

desired_tests = ["baseline unseen LOSO",
                 "MI unseen LOSO",
                 "VI unseen LOSO",
]

tests= [test for test in tests if any(desired == test for desired in desired_tests)]

remap_labels = {
    "baseline all subjects tuned": "Baseline",
    "MI all subjects": "MI-Pretrained",
    "VI all subjects": "VEP-Pretrained",
    "baseline unseen LOSO": "Baseline",
    "MI unseen LOSO": "MI-Pretrained",
    "VI unseen LOSO": "VEP-Pretrained",
}

labels = []

per_subj_accs = []

test_data = {}

np.random.seed(42)


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given data set.
    """
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_value = t.ppf((1 + confidence) / 2., n - 1)  # Two-tailed t-test
    interval = t_value * std_err
    lower_bound = mean - interval
    upper_bound = mean + interval
    return mean, (lower_bound, upper_bound)

# Plot bar chart showing number of epochs to converge and standard deviation
for test in tests:
    #if 'unseen' in test:
    #    continue
    if 'no overlap' in test:
        continue


    test_path = os.path.join(test_dir, test)
    results = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f)) and f.endswith('.json')]
    print(f"Found {len(results)} results files in {test_path}")

    if len(results) == 0:
        print(f"No results found in {test_path}")
        continue

    epochs = []
    test_f1s = []

    preds = []
    trues = []
    subj_ids = []

    for result in results:
        if 'results' in result:
            with open(os.path.join(test_path, result), 'r') as f:
                data = json.load(f)

            # Calculate accuracy from the data
            pred_np = np.array(data['pred'])
            true_np = np.array(data['true'])
            subj_id_np = np.array(data['subject'])

            preds.append(pred_np)
            trues.append(true_np)
            subj_ids.append(subj_id_np)


            continue
        if 'hyper' in result:
            continue
        print(f"Loading {result}")
        with open(os.path.join(test_path, result), 'r') as f:
            data = json.load(f)
            epochs.append(len(data['train_loss']))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    subj_ids = np.concatenate(subj_ids)

    test_data[remap_labels[test] if test in remap_labels else test] = {
        'preds': preds,
        'trues': trues,
        'subj_ids': subj_ids
    }

    

    epoch_data = calculate_confidence_interval(epochs)
    # Round data to 0 decimal places
    labels.append(remap_labels[test] if test in remap_labels else test)

# Calculate peformance drop between baseline and both models

n_bootstrap = 5000
bootstrap_MI_accs = []
bootstrap_VEP_accs = []
bootstrap_baseline_accs = []

baseline_data = test_data["Baseline"]
MI_data = test_data["MI-Pretrained"]
VEP_data = test_data["VEP-Pretrained"]

for _ in range(n_bootstrap):
    baseline_indices = np.random.choice(len(baseline_data['preds']), len(baseline_data['preds']), replace=True)
    MI_indices = np.random.choice(len(MI_data['preds']), len(MI_data['preds']), replace=True)
    VEP_indices = np.random.choice(len(VEP_data['preds']), len(VEP_data['preds']), replace=True)

    baseline_pred_bootstrap = baseline_data['preds'][baseline_indices]
    baseline_true_bootstrap = baseline_data['trues'][baseline_indices]
    MI_pred_bootstrap = MI_data['preds'][MI_indices]
    MI_true_bootstrap = MI_data['trues'][MI_indices]
    VEP_pred_bootstrap = VEP_data['preds'][VEP_indices]
    VEP_true_bootstrap = VEP_data['trues'][VEP_indices]

    baseline_acc = accuracy_score(baseline_true_bootstrap, baseline_pred_bootstrap)
    MI_acc = accuracy_score(MI_true_bootstrap, MI_pred_bootstrap)
    VEP_acc = accuracy_score(VEP_true_bootstrap, VEP_pred_bootstrap)
    #bootstrap_MI_accs.append(baseline_acc - MI_acc)
    #bootstrap_VEP_accs.append(baseline_acc - VEP_acc)

    bootstrap_baseline_accs.append(baseline_acc - 0.05) #difference from chance level
    bootstrap_MI_accs.append(MI_acc - 0.05)
    bootstrap_VEP_accs.append(VEP_acc - 0.05)

def bootstrap_ci(bootstrap_data, sig_figs=3, confidence=0.95):
    lower_bound = np.percentile(bootstrap_data, 100 * (1 - confidence) / 2)
    upper_bound = np.percentile(bootstrap_data, 100 * (1 + confidence) / 2)
    mean_acc = np.mean(bootstrap_data)
    lower_bound = np.round(lower_bound, sig_figs)
    upper_bound = np.round(upper_bound, sig_figs)
    mean_acc = np.round(mean_acc, sig_figs)

    return mean_acc, lower_bound, upper_bound


mean_baseline, lower_bound_baseline, upper_bound_baseline = bootstrap_ci(bootstrap_baseline_accs, sig_figs=3)
mean_MI, lower_bound_MI, upper_bound_MI = bootstrap_ci(bootstrap_MI_accs, sig_figs=3)
mean_VEP, lower_bound_VEP, upper_bound_VEP = bootstrap_ci(bootstrap_VEP_accs, sig_figs=3)


print("\n\nModel: {}".format("Baseline"))
print(f"Bootstrap accuracy: {mean_baseline} ({lower_bound_baseline}, {upper_bound_baseline})\n")
print("\n\nModel: {}".format("MI-Pretrained"))
print(f"Bootstrap accuracy: {mean_MI} ({lower_bound_MI}, {upper_bound_MI})\n")
print("\n\nModel: {}".format("VEP-Pretrained"))
print(f"Bootstrap accuracy: {mean_VEP} ({lower_bound_VEP}, {upper_bound_VEP})\n")








