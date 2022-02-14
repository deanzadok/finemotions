import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (8,3)

NUM_EXPS = 5
GRAPH_FORMAT = 'pdf'
# GRAPH_TITLE = 'Piano Playing'
# GRAPH_FILE = 'piano_playing'
GRAPH_TITLE = 'Keyboard Typing'
GRAPH_FILE = 'keyboard_typing'
SUBJECTS_NAMES = ['01', '04', '05', '06', '10', '17', '19', '20', '21', '23', '24', '25']

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


# append all k-fold experiments folders for all subjects
rec_dict, pre_dict = {x:[] for x in SUBJECTS_NAMES}, {x:[] for x in SUBJECTS_NAMES}
for i, exp_dir in enumerate(exp_folders):
    for j, subject_name in enumerate(SUBJECTS_NAMES):
        subject_metric_df = pd.read_csv(os.path.join(exp_dir, f'metrics_full_df_r{subject_name}.csv'))

        rec_dict[subject_name].append(subject_metric_df['rec'].dropna().values)
        pre_dict[subject_name].append(subject_metric_df['pre'].dropna().values)

# compute mean and std for each of the subjects
for j, subject_name in enumerate(SUBJECTS_NAMES):

    rec_dict[subject_name] = [np.concatenate(rec_dict[subject_name]).mean(), np.concatenate(rec_dict[subject_name]).std()]
    pre_dict[subject_name] = [np.concatenate(pre_dict[subject_name]).mean(), np.concatenate(pre_dict[subject_name]).std()]


# plot graph
eps = 5e-2
for j, subject_name in enumerate(SUBJECTS_NAMES):
    # plot recall
    plt.scatter(j, rec_dict[subject_name][0], marker='^', s=80, color='gold')
    lower_bound = max(0, rec_dict[subject_name][0] - rec_dict[subject_name][1])
    upper_bound = min(1, rec_dict[subject_name][0] + rec_dict[subject_name][1])
    plt.plot([j,j], [lower_bound, upper_bound], color='gold')
    plt.text(j+.04, rec_dict[subject_name][0]+.03, str(round(rec_dict[subject_name][0], 3)), fontsize=12, color='darkgoldenrod')

    # plot precision
    plt.scatter(j + eps, pre_dict[subject_name][0], marker='v', s=80, color='cornflowerblue')
    lower_bound = max(0, pre_dict[subject_name][0] - pre_dict[subject_name][1])
    upper_bound = min(1, pre_dict[subject_name][0] + pre_dict[subject_name][1])
    plt.plot([j + eps,j + eps], [lower_bound, upper_bound], color='cornflowerblue')
    plt.text(j+.04, pre_dict[subject_name][0]-.07, str(round(pre_dict[subject_name][0], 3)), fontsize=12, color='royalblue')

colors = ['gold', 'cornflowerblue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
labels = ['Recall', 'Precision']
plt.legend(lines, labels, loc='lower right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(range(len(SUBJECTS_NAMES)), range(1, len(SUBJECTS_NAMES)+1))
plt.title(GRAPH_TITLE)
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_subjects_cigraph_{}.{}".format(GRAPH_FILE, GRAPH_FORMAT)))
