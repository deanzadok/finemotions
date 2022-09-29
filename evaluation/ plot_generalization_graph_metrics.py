import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (5,4)
NUM_METRICS = 4
METRICS_NAMES = ['Accuracy', 'Recall', 'Precision', 'F1']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue', 'coral']
METRICS_SYMBOLS = ['o', '^', 'v', 'd']
NUM_ENROLLMENTS = 10
NUM_EXPS = 10
TARGET_EPOCH = 9
GRAPH_FORMAT = 'pdf'
EXPS_DIR="/mnt/walkure_public/username/models/mfm/us2conf2multimidi_r00_gen/mfmunet_224res_8imgs_calib_r00_multiplaying"

# (2 * pre_val * rec_val) / (pre_val + rec_val)
# load test metrics from experiments directories
metrics_values = np.zeros((NUM_METRICS, NUM_ENROLLMENTS, NUM_EXPS))
metrics_features = np.zeros((NUM_METRICS, NUM_ENROLLMENTS, 2))

# get mean values
for i in range(NUM_ENROLLMENTS):
    for j in range(NUM_EXPS):
        # load metrics csv
        metrics_df = pd.read_csv(os.path.join(EXPS_DIR + "_ne" + str(i+1) + "_s" + str(j+1), 'metrics_full_df.csv'))

        # append accuracy, recall, precision and f1
        metrics_values[0, i, j] = metrics_df['acc'].dropna().values.mean()
        metrics_values[1, i, j] = metrics_df['rec'].dropna().values.mean()
        metrics_values[2, i, j] = metrics_df['pre'].dropna().values.mean()
        metrics_values[3, i, j] = (2 * metrics_values[1, i, j] * metrics_values[2, i, j]) / (metrics_values[1, i, j] + metrics_values[2, i, j])

# prepare graph values
for i in range(NUM_METRICS):
    metrics_features[i, :, 0] = np.nanmean(np.array(metrics_values[i, :, :]), axis=1)
    metrics_features[i, :, 1] = np.nanstd(np.array(metrics_values[i, :, :]), axis=1)

# plot graph
eps = 2e-2
offsets = [0, -1*eps, 0, 1*eps]
for i in range(NUM_METRICS):
    plt.scatter(range(1, NUM_ENROLLMENTS+1), metrics_features[i, :, 0], marker=METRICS_SYMBOLS[i], s=80, color=COLORS[i])
    plt.plot(range(1, NUM_ENROLLMENTS+1), metrics_features[i, :, 0], color=COLORS[i])

    # plot std
    lower_bound = np.clip(metrics_features[i, :, 0] - metrics_features[i, :, 1], 0, 1)
    upper_bound = np.clip(metrics_features[i, :, 0] + metrics_features[i, :, 1], 0, 1)
    for j in range(NUM_ENROLLMENTS):
        plt.plot([j+1+offsets[i]]*2, [lower_bound[j], upper_bound[j]], color=COLORS[i], alpha=0.5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, marker=y, label=z, linewidth=3, markersize=8, linestyle='-') for x,y,z in zip(COLORS, METRICS_SYMBOLS,METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower right', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(range(1, NUM_ENROLLMENTS+1, 1), fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlim([1, NUM_ENROLLMENTS+1])
plt.xlabel('Enrollments')
#plt.ylabel(r"$\mathcal{L}_{BCE}$")

# save plot
plt.grid(True)
plt.tight_layout(pad=0.7)
plt.savefig("generalization_graph_metrics.{}".format(GRAPH_FORMAT))