import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (3.5,3)

METRIC_IDX = 3
NUM_EPOCHS = 20
NUM_EXPS = 5
GRAPH_FORMAT = 'pdf'
# GRAPH_TITLE = 'Piano Playing'
# GRAPH_FILE = 'piano_playing'
GRAPH_TITLE = 'Keyboard Typing'
GRAPH_FILE = 'keyboard_typing'
METHODS_NAMES = ['SF', 'MF', 'CBMF']
NUM_METHODS = len(METHODS_NAMES)
METRICS_NAMES = ['Accuracy', 'Recall', 'Precision', 'F1']
METRICS_COLUMNS = ['acc', 'rec', 'pre', 'f1']

if GRAPH_FILE == 'piano_playing':
    methods_folders = [['/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_01_st0.8_kf4',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_02_st0.8_kf0',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_03_st0.8_kf1',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_04_st0.8_kf2',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_05_st0.8_kf3'],
                        ['/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_03_st0.8_sequence_kf4', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_04_st0.8_sequence_kf1', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_05_st0.8_sequence_kf0', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_06_st0.8_sequence_kf2', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_07_st0.8_sequence_kf3'],
                        ['/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']]

else:
    methods_folders = [['/mnt/walkure_public/deanz/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_01_st0.8_kf4',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_02_st0.8_kf0',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_03_st0.8_kf1',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_04_st0.8_kf2',
                        '/mnt/walkure_public/deanz/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_05_st0.8_kf3'],
                        ['/mnt/walkure_public/deanz/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_05_st0.8_sequence_kf4', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_06_st0.8_sequence_kf1', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_07_st0.8_sequence_kf0', 
                        '/mnt/walkure_public/deanz/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_08_st0.8_sequence_kf2',
                        '/mnt/walkure_public/deanz/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_09_st0.8_sequence_kf3'],
                        ['/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                        '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']]

methods_dfs = {}
for i, method_exps in enumerate(methods_folders):
    # append all k-fold xperiments into one dataframe
    method_df = pd.DataFrame(columns=['acc','rec','pre','f1'])
    for j, exp_dir in enumerate(method_exps):
        method_df = method_df.append(pd.read_csv(os.path.join(exp_dir, 'metrics_full_df.csv')))

    # store it
    methods_dfs[METHODS_NAMES[i]] = method_df

methods_means, methods_stds = {}, {}
for i, method_name in enumerate(METHODS_NAMES):
    if METRIC_IDX == 3:
        methods_means[method_name] = 2 * (methods_dfs[method_name]['rec'].dropna().mean() * methods_dfs[method_name]['pre'].dropna().mean()) / (methods_dfs[method_name]['rec'].dropna().mean() + methods_dfs[method_name]['pre'].dropna().mean())
        methods_stds[method_name] = 2 * (methods_dfs[method_name]['rec'].dropna().std() * methods_dfs[method_name]['pre'].dropna().std()) / (methods_dfs[method_name]['rec'].dropna().std() + methods_dfs[method_name]['pre'].dropna().std())
    else:
        methods_means[method_name] = methods_dfs[method_name][METRICS_COLUMNS[METRIC_IDX]].dropna().mean()
        methods_stds[method_name] = methods_dfs[method_name][METRICS_COLUMNS[METRIC_IDX]].dropna().std()
    
# plot graph
for i, method_name in enumerate(METHODS_NAMES):
    plt.scatter(i, methods_means[method_name], marker='d', s=80)
    plt.text(i, methods_means[method_name]+.03, str(round(methods_means[method_name], 3)), fontsize=14)

    lower_bound = max(0, methods_means[method_name] - methods_stds[method_name])
    upper_bound = min(1, methods_means[method_name] + methods_stds[method_name])
    plt.plot([i,i], [lower_bound, upper_bound])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(range(NUM_METHODS), METHODS_NAMES)
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_cigraph_{}_{}.{}".format(GRAPH_FILE, METRICS_COLUMNS[METRIC_IDX], GRAPH_FORMAT)))
