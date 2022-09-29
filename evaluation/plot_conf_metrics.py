import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (17,2.3)
JOINT_NAMES = ['joint41', 'joint31', 'joint21', 'joint11', 'joint42', 'joint32',
               'joint22', 'joint12', 'joint43', 'joint33', 'joint23', 'joint13',
               'jointt5', 'jointt6', 'jointwr', 'jointwp', 'jointwy']
COLORS = ['orange', 'slateblue']
LABELS = ['K. Typing', 'P. Playing']
# JOINT_NAMES_GRAPH = ['j41',       'j31',      'j21',      'j11',     'j42',      'j32',     'j22',      'j12',       'j43',     'j33',       'j23',      'j13',     'jt5',       'jt6',     'jwr',      'jwp',      'jwy']
JOINT_NAMES_GRAPH = [r'$J^5_1$', r'$J^4_1$', r'$J^3_1$', r'$J^2_1$', r'$J^5_2$', r'$J^4_2$',r'$J^3_2$', r'$J^2_2$', r'$J^5_3$', r'$J^4_3$', r'$J^3_3$', r'$J^2_3$', r'$J^1_1$', r'$J^1_2$', r'$J^w_r$', r'$J^w_p$', r'$J^w_y$']
GRAPH_FORMAT = 'pdf'

# define experiments folders
pia_methods_folders = ['/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']
key_methods_folders = ['/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                       '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']

# append all k-fold experiments into two ik and fk dataframes
pia_metrics_conf_ik_df = pd.DataFrame(columns=JOINT_NAMES)
key_metrics_conf_ik_df = pd.DataFrame(columns=JOINT_NAMES)

# load errors from experiments folders
#metrics_conf_fk_df = pd.DataFrame(columns=['finger4', 'finger3', 'finger2', 'finger1', 'thumb'])
pia_metrics_conf_ik_means = pia_metrics_conf_ik_df.copy()
pia_metrics_conf_ik_stds = pia_metrics_conf_ik_df.copy()
key_metrics_conf_ik_means = key_metrics_conf_ik_df.copy()
key_metrics_conf_ik_stds = key_metrics_conf_ik_df.copy()
# for i, method_dir in enumerate(pia_methods_folders):
#     pia_metrics_conf_ik_df = pia_metrics_conf_ik_df.append(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')))
# for i, method_dir in enumerate(key_methods_folders):
#     key_metrics_conf_ik_df = key_metrics_conf_ik_df.append(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')))
pia_metrics_conf_ik_means, key_metrics_conf_ik_means = [], []
for i, method_dir in enumerate(pia_methods_folders):
    pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')).mean()
    pia_metrics_conf_ik_means.append(np.expand_dims(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')).mean().values, axis=0))
for i, method_dir in enumerate(key_methods_folders):
    key_metrics_conf_ik_means.append(np.expand_dims(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')).mean().values, axis=0))
pia_metrics_conf_ik_means = np.sqrt(np.concatenate(pia_metrics_conf_ik_means, axis=0))
key_metrics_conf_ik_means = np.sqrt(np.concatenate(key_metrics_conf_ik_means, axis=0))

# compute mean and std
pia_metrics_conf_ik_stds = pia_metrics_conf_ik_means.std(axis=0).round(3)
key_metrics_conf_ik_stds = key_metrics_conf_ik_means.std(axis=0).round(3)
pia_metrics_conf_ik_means = pia_metrics_conf_ik_means.mean(axis=0).round(3)
key_metrics_conf_ik_means = key_metrics_conf_ik_means.mean(axis=0).round(3)

# put means and stds into dataframes and store
# pia_metrics_conf_ik_df = pia_metrics_conf_ik_df.applymap(lambda x: np.sqrt(x))
# key_metrics_conf_ik_df = key_metrics_conf_ik_df.applymap(lambda x: np.sqrt(x))
# pia_metrics_conf_ik_means = pia_metrics_conf_ik_means.append(pia_metrics_conf_ik_df.mean(), ignore_index=True).round(3)
# key_metrics_conf_ik_means = key_metrics_conf_ik_means.append(key_metrics_conf_ik_df.mean(), ignore_index=True).round(3)
# pia_metrics_conf_ik_stds = pia_metrics_conf_ik_stds.append(pia_metrics_conf_ik_df.std(), ignore_index=True).round(3)
# key_metrics_conf_ik_stds = key_metrics_conf_ik_stds.append(key_metrics_conf_ik_df.std(), ignore_index=True).round(3)

# plot graph
eps = 1e-1
y_start = 0.015
for j, joint_name in enumerate(JOINT_NAMES):

    # plot keyboard value
    plt.scatter(j, key_metrics_conf_ik_means[j], marker='*', s=100, color=COLORS[0])
    plt.plot([j,j], [key_metrics_conf_ik_means[j] - key_metrics_conf_ik_stds[j], key_metrics_conf_ik_means[j] + key_metrics_conf_ik_stds[j]], color='orange')
    text_loc_x = j + 0.1
    text_loc_y = y_start + 0.015
    plt.text(text_loc_x, text_loc_y, str(round(key_metrics_conf_ik_means[j], 3)), fontsize=FONT_SIZE-2, color='darkorange')

    # plot piano values
    plt.scatter(j + eps, pia_metrics_conf_ik_means[j], marker='*', s=100, color=COLORS[1])
    plt.plot([j + eps,j + eps], [pia_metrics_conf_ik_means[j] - pia_metrics_conf_ik_stds[j], pia_metrics_conf_ik_means[j] + pia_metrics_conf_ik_stds[j]], color='slateblue')
    text_loc_y = y_start
    plt.text(text_loc_x, text_loc_y, str(round(pia_metrics_conf_ik_means[j], 3)), fontsize=FONT_SIZE-2, color='darkslateblue')

# prepare customized legend
lines = [Line2D([0], [0], color=c, marker='*', label=y, linewidth=3, markersize=12, linestyle='-') for c,y in zip(COLORS, LABELS)]
plt.legend(handles=lines, loc='upper right', fontsize=FONT_SIZE-2)

# prepare axes
plt.xticks(range(len(JOINT_NAMES)), JOINT_NAMES_GRAPH, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Joint', fontsize=FONT_SIZE-3)
plt.ylabel("Error [rad]", fontsize=FONT_SIZE-3)
plt.xlim([-0.2, 18.5])
plt.ylim([y_start - 0.002, 0.115])

# save plot
plt.grid(True)
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join("metrics_confs_cigraph.{}".format(GRAPH_FORMAT)))
