from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from scipy.stats import mannwhitneyu

dataroots = {
    'PROPOSAL': 'results/main_manual',
    'NOSPECLR': 'results/main_manual_abl_layerlr1e-4',
    'NOROT': 'results/main_manual_abl_testrot',
}


def get_values(dataroot, key):
    npzs = list(glob(join(dataroot, '*', 'results.npz')))
    vals = []
    for f in npzs:
        recorded_data = np.load(f)
        val = recorded_data[key]
        vals.append(val)
    vals = np.stack(vals, 0)
    return vals


def plot_mean_std(dataroot, key, ax, **kwargs):
    vals = get_values(dataroot, key)
    mean = np.mean(vals, 0)
    std = np.std(vals, 0)
    epochs = np.arange(len(mean))

    # Offset by 1 so that we have nicely zoomed plots
    mean = mean[1:]
    std = std[1:]
    epochs = epochs[1:]

    ax.plot(epochs, mean, **kwargs)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)


def plot3(key, ax):
    for k, v in dataroots.items():
        plot_mean_std(v, key, ax, label=k)


def print_final_min_mean_max(dataroot, key, model_epochs):
    vals = get_values(dataroot, key) * 100
    vals = vals[np.arange(len(vals)), model_epochs]
    min = np.min(vals)
    mean = np.mean(vals)
    std = np.std(vals)
    max = np.max(vals)
    print(f'{min:.2f}', f'{mean:.2f} ± {std:.2f}', f'{max:.2f}', sep='\t')


def print_final_table(dataroot):
    best_model_epochs = np.argmax(get_values(dataroot, 'f1'), axis=1)

    print_final_min_mean_max(dataroot, 'acc', best_model_epochs)
    print_final_min_mean_max(dataroot, 'acc_all', best_model_epochs)
    print_final_min_mean_max(dataroot, 'acc_hem', best_model_epochs)
    print_final_min_mean_max(dataroot, 'f1', best_model_epochs)
    print_final_min_mean_max(dataroot, 'precision', best_model_epochs)
    print_final_min_mean_max(dataroot, 'recall', best_model_epochs)


def get_best_f1_scores(dataroot):
    f1_scores = get_values(dataroot, 'f1')
    best_model_epochs = np.argmax(f1_scores, axis=1)
    return f1_scores[np.arange(len(f1_scores)), best_model_epochs]


def is_statistically_greater(dataroot1, dataroot2):
    # Tests if F1-score of dataroot1 is greater than dataroot2
    a = get_best_f1_scores(dataroot1)
    b = get_best_f1_scores(dataroot2)
    u, p = mannwhitneyu(a, b, alternative='greater')
    return u, p


######

for k, v in dataroots.items():
    print(k)
    print_final_table(v)
    print()


######

print("MWU-Test of PROPOSAL > NOSPECLR")
print(is_statistically_greater(dataroots['PROPOSAL'], dataroots['NOSPECLR']))
print()
print("MWU-Test of PROPOSAL > NOROT")
print(is_statistically_greater(dataroots['PROPOSAL'], dataroots['NOROT']))

######

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))

ax[0, 0].set_title('Accuracy')
plot3('acc', ax[0, 0])

ax[0, 1].set_title('Sensitivity')
plot3('acc_all', ax[0, 1])

ax[0, 2].set_title('Specificity')
plot3('acc_hem', ax[0, 2])

ax[1, 0].set_title('F1 score')
plot3('f1', ax[1, 0])

ax[1, 1].set_title('Precision')
plot3('precision', ax[1, 1])

ax[1, 2].set_title('Recall')
plot3('recall', ax[1, 2])

fig.legend(loc='lower center', ncol=3)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
fig.savefig('results/plot_ablations.pdf')

######

subj_acc = np.load('results/main_manual/20190313T101236Z.LGJL/subj_acc.npz')
subj = list(sorted(subj_acc.keys()))
acc = [subj_acc[k] for k in subj]
fig, ax = plt.subplots(figsize=(9, 2))
ax.bar(range(len(acc)), acc, width=0.3, tick_label=subj)
fig.tight_layout()
fig.savefig('results/plot_subj_acc.pdf')

######

data = np.load('results/main_manual/20190313T101236Z.LGJL/results.npz')
loss_train = data['loss_train']
loss_valid = data['loss_valid'][1:]
f1_valid = data['f1'][1:]
fig, ax = plt.subplots(ncols=3, figsize=(9, 2))
ax[0].plot(range(len(loss_train)), loss_train)
ax[0].set_title("Training set loss")
ax[1].plot(range(1, len(loss_valid) + 1), loss_valid)
ax[1].set_title("Preliminary test set loss")
ax[2].plot(range(1, len(f1_valid) + 1), f1_valid)
ax[2].set_title("Preliminary test set F1-score")
fig.tight_layout()
fig.savefig('results/plot_curves.pdf')

######

plt.show()
