import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 15
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (13.5,2.0)
METRIC_IDX = 3
NUM_TASKS = 2
NUM_EXPS = 5
TASKS_NAMES = ['Piano Playing', 'Keyboard Typing']
METHODS_NAMES = ['[12]+MLP', '[9]+MLP', 'SF', 'MF', 'CBMF']
METHODS_COLORS = ['brown', 'indianred', 'goldenrod', 'deepskyblue', 'rebeccapurple']
NUM_METHODS = len(METHODS_NAMES)
METRICS_NAMES = ['Accuracy', 'Recall', 'Precision', 'F1']
METRICS_SYMBOLS = ['o', '^', 'v', 'd']
METRICS_COLUMNS = ['acc', 'rec', 'pre', 'f1']
GRAPH_FORMAT = 'pdf'

# folders to all experiments
experiments_dirs = [[['/mnt/walkure_public/username/models/classic/us2multimidi_all/perfor_224res_8imgs_calib_all_multiplaying_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/perfor_224res_8imgs_calib_all_multiplaying_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/perfor_224res_8imgs_calib_all_multiplaying_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/perfor_224res_8imgs_calib_all_multiplaying_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/perfor_224res_8imgs_calib_all_multiplaying_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/classic/us2multimidi_all/echoflex_224res_8imgs_calib_all_multiplaying_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/echoflex_224res_8imgs_calib_all_multiplaying_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/echoflex_224res_8imgs_calib_all_multiplaying_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/echoflex_224res_8imgs_calib_all_multiplaying_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/classic/us2multimidi_all/echoflex_224res_8imgs_calib_all_multiplaying_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_03_st0.8_sequence_kf4',
                      '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_04_st0.8_sequence_kf1',
                      '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_05_st0.8_sequence_kf0',
                      '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_06_st0.8_sequence_kf2',
                      '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_07_st0.8_sequence_kf3'],
                     ['/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']],
                    [['/mnt/walkure_public/username/models/classic/us2multikey_all/perfor_224res_8imgs_calib_all_multityping_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/perfor_224res_8imgs_calib_all_multityping_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/perfor_224res_8imgs_calib_all_multityping_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/perfor_224res_8imgs_calib_all_multityping_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/perfor_224res_8imgs_calib_all_multityping_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/classic/us2multikey_all/echoflex_224res_8imgs_calib_all_multityping_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/echoflex_224res_8imgs_calib_all_multityping_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/echoflex_224res_8imgs_calib_all_multityping_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/echoflex_224res_8imgs_calib_all_multityping_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/classic/us2multikey_all/echoflex_224res_8imgs_calib_all_multityping_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_01_st0.8_kf4',
                      '/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_02_st0.8_kf0',
                      '/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_03_st0.8_kf1',
                      '/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_04_st0.8_kf2',
                      '/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_05_st0.8_kf3'],
                     ['/mnt/walkure_public/username/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_05_st0.8_sequence_kf4', 
                      '/mnt/walkure_public/username/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_06_st0.8_sequence_kf1', 
                      '/mnt/walkure_public/username/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_07_st0.8_sequence_kf0', 
                      '/mnt/walkure_public/username/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_08_st0.8_sequence_kf2',
                      '/mnt/walkure_public/username/models/mfm/us2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_09_st0.8_sequence_kf3'],
                     ['/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                      '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']]]

# load csvs with all four metrics for all experiments
tasks_dfs = {}
for i, task_name in enumerate(TASKS_NAMES):
    methods_dfs = {}
    for j, method_name in enumerate(METHODS_NAMES):
        
        # append all k-fold xperiments into one dataframe
        method_metrics = []
        for k in range(NUM_EXPS):

            # metrics were already averaged for classic methods
            if os.path.isfile(os.path.join(experiments_dirs[i][j][k], 'metrics_full_df_mean.csv')):
                metrics_df = pd.read_csv(os.path.join(experiments_dirs[i][j][k], 'metrics_full_df_mean.csv'))
                metrics_np = metrics_df.values[0]
            else:
                metrics_df = pd.read_csv(os.path.join(experiments_dirs[i][j][k], 'metrics_full_df.csv'))

                # compute mean value of metrics
                metrics_np = np.zeros(4)
                for l in range(3):
                    metrics_np[l] = metrics_df[METRICS_COLUMNS[l]].dropna().mean()
                metrics_np[-1] = (2 * metrics_np[1] * metrics_np[2]) / (metrics_np[1] + metrics_np[2])

            # append
            method_metrics.append(np.expand_dims(metrics_np, axis=0))

        # store the methods experiments
        methods_dfs[method_name] = np.concatenate(method_metrics, axis=0)
    
    # store the experiments according to the task
    tasks_dfs[task_name] = methods_dfs

# calculate mean and std values of all experiments
tasks_features = {}
for i, task_name in enumerate(TASKS_NAMES):
    methods_features = {}
    for j, method_name in enumerate(METHODS_NAMES):
        metrics_features = {}
        for k, metric_column in enumerate(METRICS_COLUMNS):

            metrics_features[metric_column] = [np.nanmean(tasks_dfs[task_name][method_name][:,k]), np.nanstd(tasks_dfs[task_name][method_name][:,k])]

        # append all metrics to method
        methods_features[method_name] = metrics_features

    # append all methods to task
    tasks_features[task_name] = methods_features


# plot graph
y_start = 0.1
step_size = 1
eps = 8e-2
x = step_size - eps - 0.15
x_text = step_size
for i, task_name in enumerate(TASKS_NAMES):
    for k, metric_column in enumerate(METRICS_COLUMNS):
        for j, method_name in enumerate(METHODS_NAMES):
            
            metric_features = tasks_features[task_name][method_name][metric_column]
            if metric_features[0] > 0:
                plt.scatter(x, metric_features[0], marker=METRICS_SYMBOLS[k], color=METHODS_COLORS[j] , s=64)
            text_loc_y = y_start + 0.01 + j * 0.11
            if metric_features[0] >= 0 and metric_features[0] <= 1:
                plt.text(x_text+0.07, text_loc_y, str(round(metric_features[0], 3)), color=METHODS_COLORS[j], fontsize=FONT_SIZE-1)
                
                lower_bound = max(0, metric_features[0] - metric_features[1])
                upper_bound = min(1, metric_features[0] + metric_features[1])
                plt.plot([x]*2, [lower_bound, upper_bound], color=METHODS_COLORS[j])
            else:
                plt.text(x_text+0.07, text_loc_y, 'N/A', color=METHODS_COLORS[j], fontsize=FONT_SIZE-1)


            x += eps
        x += (step_size - NUM_METHODS*eps)
        x_text += step_size

# separate between the two tasks
plt.plot([4.6]*2, [-0.05, 1.05], color='dimgrey')

# prepare customized legend
legend_lines = [Line2D([0], [0], color=x, label=y, linewidth=3, linestyle='-') for x,y in zip(METHODS_COLORS,METHODS_NAMES)]
legend_symbols = [Line2D([], [], color='black', marker=x, linestyle='None', markersize=7, label=y) for x,y in zip(METRICS_SYMBOLS, METRICS_NAMES)]
metrics_legend = plt.legend(handles=legend_lines+legend_symbols, loc='lower right', fontsize=FONT_SIZE-7)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.tick_params(left = False, bottom = False)
plt.xticks(range(1, len(METRICS_NAMES)*2 + 1), METRICS_NAMES*2, fontsize=FONT_SIZE-3)
plt.yticks(np.arange(0,1.25, step=0.25), fontsize=FONT_SIZE)
plt.xlim([0.7, 9.2])
plt.ylim([y_start, 1.0])

# prepare title
title_spaces = [' ']*53
end_spaces = [' ']*14
plt.title(TASKS_NAMES[0] + ''.join(title_spaces) + TASKS_NAMES[1] + ''.join(end_spaces), loc='center')

# save plot
plt.grid(True, which='major')
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_cigraph_combined.{}".format(GRAPH_FORMAT)))