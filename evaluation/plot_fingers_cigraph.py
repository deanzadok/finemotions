import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (3.5,3)

GRAPH_FORMAT = 'pdf'
# GRAPH_TITLE = 'Piano Playing'
# GRAPH_FILE = 'piano_playing'
GRAPH_TITLE = 'Keyboard Typing'
GRAPH_FILE = 'keyboard_typing'

CLASSES_IDXS = ['1', '2', '3', '4', '5']
CLASSES_NAMES = [r'$F^1$', r'$F^2$', r'$F^3$', r'$F^4$', r'$F^5$']

if GRAPH_FILE == 'piano_playing':
    exp_folders = ['/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']
else:
    exp_folders = ['/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']

# append all k-fold experiments folders for all classes
rec_dict, pre_dict = {x:[] for x in CLASSES_IDXS}, {x:[] for x in CLASSES_IDXS}
for i, exp_dir in enumerate(exp_folders):

    rec_df = pd.read_csv(os.path.join(exp_dir, 'metrics_classes_rec.csv'))
    pre_df = pd.read_csv(os.path.join(exp_dir, 'metrics_classes_pre.csv'))

    for class_idx in CLASSES_IDXS:
        rec_dict[class_idx].append(rec_df[class_idx].dropna().values)
        pre_dict[class_idx].append(pre_df[class_idx].dropna().values)


# compute mean and std for each of the classes
for j, class_idx in enumerate(CLASSES_IDXS):

    rec_dict[class_idx] = [np.concatenate(rec_dict[class_idx]).mean(), np.concatenate(rec_dict[class_idx]).std()]
    pre_dict[class_idx] = [np.concatenate(pre_dict[class_idx]).mean(), np.concatenate(pre_dict[class_idx]).std()]


# plot graph
eps = 5e-2
for j, class_idx in enumerate(CLASSES_IDXS):
    # plot recall
    plt.scatter(j, rec_dict[class_idx][0], marker='^', s=80, color='gold')
    lower_bound = max(0, rec_dict[class_idx][0] - rec_dict[class_idx][1])
    upper_bound = min(1, rec_dict[class_idx][0] + rec_dict[class_idx][1])
    plt.plot([j,j], [lower_bound, upper_bound], color='gold')
    plt.text(j+.04, rec_dict[class_idx][0]+.03, str(round(rec_dict[class_idx][0], 3)), fontsize=12, color='darkgoldenrod')

    # plot precision]
    plt.scatter(j + eps, pre_dict[class_idx][0], marker='v', s=80, color='cornflowerblue')
    lower_bound = max(0, pre_dict[class_idx][0] - pre_dict[class_idx][1])
    upper_bound = min(1, pre_dict[class_idx][0] + pre_dict[class_idx][1])
    plt.plot([j + eps,j + eps], [lower_bound, upper_bound], color='cornflowerblue')
    plt.text(j+.04, pre_dict[class_idx][0]-.07, str(round(pre_dict[class_idx][0], 3)), fontsize=12, color='royalblue')



colors = ['gold', 'cornflowerblue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
labels = ['Recall', 'Precision']
plt.legend(lines, labels)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(range(len(CLASSES_IDXS)), CLASSES_NAMES)
plt.title(GRAPH_TITLE)
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_classes_cigraph_{}.{}".format(GRAPH_FILE, GRAPH_FORMAT)))