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

class2lab = {key: value-1 for key, value in class2lab.items()}

reverselab = {v: k for k, v in class2lab.items()}


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

    preds = np.array([reverselab[p] for p in preds])
    trues = np.array([reverselab[t] for t in trues])

    # Create confusion matrix
    class_labels = np.unique(trues)
    cm = confusion_matrix(trues, preds, labels=class_labels)
    # Normalise confusion matrix
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Round to 2 decimal places
    #cm = np.round(cm, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    # Set borders of plot
    plot = disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, values_format='.0f')
    for text in plot.ax_.texts:
        text.set_fontsize(7)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.2)
    plt.show()




    """cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_labels)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=200, nrows=1, ncols=2)

    plot = disp.plot(cmap=plt.cm.Blues, ax=ax[0], colorbar=False, values_format='.0f')
    for text in plot.ax_.texts:
        text.set_fontsize(6)


    plot = disp_normalized.plot(cmap=plt.cm.Blues, ax=ax[1], values_format='.2f', colorbar=False)
    im = plot.im_
    for text in plot.ax_.texts:
        text.set_fontsize(4)

    # Get minimum and maximum values of the confusion matrix
    vmin = cm_normalized.min()
    vmax = cm_normalized.max()

    # Draw colourbar
    cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.8)
    cbar.set_label('Proportion of samples (%)')
    # generate 4 ticks from 0 to max
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.set_ticklabels([f'{int(x * 100)}%' for x in np.linspace(vmin, vmax, 5)])

    ax[0].set_ylabel('True Label')
    # Remove ax 2 y title
    ax[1].set_ylabel('')
    for axid in range(2):
        ax[axid].set_xlabel('Predicted Label')
        for tick in ax[axid].get_xticklabels():
            tick.set_rotation(90)


    ax[0].set_title(f"Confusion Matrix")
    ax[1].set_title(f"Normalised Confusion Matrix")
    plt.show()"""


    

    epoch_data = calculate_confidence_interval(epochs)
    # Round data to 0 decimal places
    labels.append(remap_labels[test] if test in remap_labels else test)







