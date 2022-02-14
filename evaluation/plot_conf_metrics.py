import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (16,3)


JOINT_NAMES = ['joint41', 'joint31', 'joint21', 'joint11', 'joint42', 'joint32',
               'joint22', 'joint12', 'joint43', 'joint33', 'joint23', 'joint13',
               'jointt5', 'jointt6', 'jointwr', 'jointwp', 'jointwy']
JOINT_NAMES_GRAPH = ['j41', 'j31', 'j21', 'j11', 'j42', 'j32',
                     'j22', 'j12', 'j43', 'j33', 'j23', 'j13',
                     'jt5', 'jt6', 'jwr', 'jwp', 'jwy']
GRAPH_FORMAT = 'pdf'

pia_methods_folders = ['/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']
key_methods_folders = ['/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                       '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']

# append all k-fold xperiments into two ik and fk dataframes
pia_metrics_conf_ik_df = pd.DataFrame(columns=JOINT_NAMES)
key_metrics_conf_ik_df = pd.DataFrame(columns=JOINT_NAMES)

#metrics_conf_fk_df = pd.DataFrame(columns=['finger4', 'finger3', 'finger2', 'finger1', 'thumb'])
pia_metrics_conf_ik_means = pia_metrics_conf_ik_df.copy()
pia_metrics_conf_ik_stds = pia_metrics_conf_ik_df.copy()
key_metrics_conf_ik_means = key_metrics_conf_ik_df.copy()
key_metrics_conf_ik_stds = key_metrics_conf_ik_df.copy()
for i, method_dir in enumerate(pia_methods_folders):
    pia_metrics_conf_ik_df = pia_metrics_conf_ik_df.append(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')))
for i, method_dir in enumerate(key_methods_folders):
    key_metrics_conf_ik_df = key_metrics_conf_ik_df.append(pd.read_csv(os.path.join(method_dir, 'metrics_conf_ik_df.csv')))

# put means and stds into dataframes and store
pia_metrics_conf_ik_df = pia_metrics_conf_ik_df.applymap(lambda x: np.sqrt(x))
key_metrics_conf_ik_df = key_metrics_conf_ik_df.applymap(lambda x: np.sqrt(x))

pia_metrics_conf_ik_means = pia_metrics_conf_ik_means.append(pia_metrics_conf_ik_df.mean(), ignore_index=True).round(3)
key_metrics_conf_ik_means = key_metrics_conf_ik_means.append(key_metrics_conf_ik_df.mean(), ignore_index=True).round(3)
pia_metrics_conf_ik_stds = pia_metrics_conf_ik_stds.append(pia_metrics_conf_ik_df.std(), ignore_index=True).round(3)
key_metrics_conf_ik_stds = key_metrics_conf_ik_stds.append(key_metrics_conf_ik_df.std(), ignore_index=True).round(3)

# plot graph
eps = 5e-2
for j, joint_name in enumerate(JOINT_NAMES):
    # plot keyboard value
    plt.scatter(j, key_metrics_conf_ik_means[joint_name][0], marker='^', s=80, color='orange')
    plt.plot([j,j], [key_metrics_conf_ik_means[joint_name][0] - key_metrics_conf_ik_stds[joint_name][0], key_metrics_conf_ik_means[joint_name][0] + key_metrics_conf_ik_stds[joint_name][0]], color='orange')
    plt.text(j-.55, key_metrics_conf_ik_means[joint_name][0]-.02, str(round(key_metrics_conf_ik_means[joint_name][0], 3)), fontsize=12, color='darkorange')

    # plot piano values
    plt.scatter(j + eps, pia_metrics_conf_ik_means[joint_name][0], marker='v', s=80, color='slateblue')
    plt.plot([j + eps,j + eps], [pia_metrics_conf_ik_means[joint_name][0] - pia_metrics_conf_ik_stds[joint_name][0], pia_metrics_conf_ik_means[joint_name][0] + pia_metrics_conf_ik_stds[joint_name][0]], color='slateblue')
    plt.text(j+.04, pia_metrics_conf_ik_means[joint_name][0]+.03, str(round(pia_metrics_conf_ik_means[joint_name][0], 3)), fontsize=12, color='darkslateblue')

colors = ['orange', 'slateblue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
labels = ['Keyboard Typing', 'Piano Playing']
plt.legend(lines, labels, loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(range(len(JOINT_NAMES)), JOINT_NAMES_GRAPH)
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_confs_cigraph.{}".format(GRAPH_FORMAT)))