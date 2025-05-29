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
per_subj_errs = []
per_subj_lower = []
per_subj_upper = []


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

    n_bootstrap = 1000

    # Calculate accuracy on a per-subject basis
    unique_subj_ids = np.unique(subj_ids)
    subject_accuracies = {}
    #subject_errs = {}
    subject_lower = {}
    subject_upper = {}
    for subj_id in unique_subj_ids:
        mask = (subj_ids == subj_id)
        pred_subj = preds[mask]
        true_subj = trues[mask]

        bootstrap_accs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(pred_subj), len(pred_subj), replace=True)
            pred_bootstrap = pred_subj[indices]
            true_bootstrap = true_subj[indices]

            bootstrap_accs.append(accuracy_score(true_bootstrap, pred_bootstrap))

        

        #acc = accuracy_score(true_subj, pred_subj)
        subject_accuracies[subj_id] = np.mean(bootstrap_accs)
        #subject_errs[subj_id] = np.mean(np.percentile(bootstrap_accs, 97.5) - np.percentile(bootstrap_accs, 2.5))
        subject_lower[subj_id] = subject_accuracies[subj_id] - np.percentile(bootstrap_accs, 2.5)
        subject_upper[subj_id] = np.percentile(bootstrap_accs, 97.5) - subject_accuracies[subj_id]

        #print(f"Subject {subj_id}: {subject_accuracies[subj_id]} ({subject_errs[subj_id]})")
        print(f"Subject {subj_id}: {subject_accuracies[subj_id]} ({subject_lower[subj_id]}, {subject_upper[subj_id]})")

    per_subj_accs.append(subject_accuracies)
    #per_subj_errs.append(subject_errs)
    per_subj_upper.append(subject_upper)
    per_subj_lower.append(subject_lower)

    

    epoch_data = calculate_confidence_interval(epochs)
    # Round data to 0 decimal places
    labels.append(remap_labels[test] if test in remap_labels else test)

participants = sorted(list(per_subj_accs[0].keys()))
n_participants = len(participants)
n_models = len(per_subj_accs)

# swap model 2 and 3 for a better visualisation
per_subj_accs[1], per_subj_accs[2] = per_subj_accs[2], per_subj_accs[1]
per_subj_lower[1], per_subj_lower[2] = per_subj_lower[2], per_subj_lower[1]
per_subj_upper[1], per_subj_upper[2] = per_subj_upper[2], per_subj_upper[1]
#per_subj_errs[1], per_subj_errs[2] = per_subj_errs[2], per_subj_errs[1]
labels[1], labels[2] = labels[2], labels[1]

x = np.arange(n_participants)  # the label locations
width = 0.25  # the width of the bars

colors = ['blue', 'orange', 'green']


fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

for i, (model_acc_dict, model_lower_dict, model_upper_dict) in enumerate(zip(per_subj_accs, per_subj_lower, per_subj_upper)):
    accuracies = [model_acc_dict[subj_id] for subj_id in participants]
    min_acc = np.min(accuracies)
    min_subj = participants[np.argmin(accuracies)]
    max_acc = np.max(accuracies)
    max_subj = participants[np.argmax(accuracies)]
    print(f"Model: {labels[i]}")
    print(f"Min accuracy: {min_acc} ({min_subj})")
    print(f"Max accuracy: {max_acc} ({max_subj})")
    #errors = [model_err_dict[subj_id] for subj_id in participants]
    lowers = [model_lower_dict[subj_id] for subj_id in participants]
    uppers = [model_upper_dict[subj_id] for subj_id in participants]
    offset = (i - n_models / 2) * width + width / 2
    ax.bar(x + offset, accuracies, width=width, label=labels[i], color=colors[i % len(colors)], zorder=2, alpha=1, yerr=[lowers, uppers], capsize=1, error_kw={'linewidth': 0.75, 'ecolor': 'black', 'alpha': 0.6})
    #ax.errorbar(x + offset, accuracies, yerr=errors, fmt='o', color='black', capsize=5, zorder=3, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(participants)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Participant')
ax.legend(loc="upper center", fontsize=10, title="Model", title_fontsize='13', frameon=False, bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.tight_layout()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=1)
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.show()





