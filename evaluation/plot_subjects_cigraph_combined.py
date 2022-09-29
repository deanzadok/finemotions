import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (11.7,2.5)
NUM_EXPS = 5
SUBJECTS_NAMES = ['01', '04', '05', '06', '10', '17', '19', '20', '21', '23', '24', '25']
TASKS_NAMES = ['Piano Playing', 'Keyboard Typing']
TASKS_NAMES_SHORT = ['P. Playing', 'K. Typing']
METRICS_NAMES = ['Recall', 'Precision']
METRICS_SYMBOLS = ['^', 'v']
METRICS_COLUMNS = ['rec', 'pre']
COLORS = ['darkturquoise', 'darkcyan', 'goldenrod', 'darkgoldenrod']
GRAPH_FORMAT = 'pdf'

exps_folders = [['/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_mp_4qloss_kf4',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_mp_4qloss_kf0',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_mp_4qloss_kf1',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_mp_4qloss_kf2',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3'],
                ['/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_23_st0.8_sequence_reslayer_retrained_mt_4qloss_kf4',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_24_st0.8_sequence_reslayer_retrained_mt_4qloss_kf0',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_25_st0.8_sequence_reslayer_retrained_mt_4qloss_kf1',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_26_st0.8_sequence_reslayer_retrained_mt_4qloss_kf2',
                 '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']]

# load csvs with all four metrics for all experiments
tasks_dfs = {}

for i, task_name in enumerate(TASKS_NAMES):
    subjects_df = {}

    # iterate over all subjects
    for j, subject_name in enumerate(SUBJECTS_NAMES):
        metrics_df = {}

        for k, metric_column in enumerate(METRICS_COLUMNS):
            metrics_df[METRICS_NAMES[k]] = []
            for l in range(NUM_EXPS):

                # load the metric values from csv
                subject_metric_df = pd.read_csv(os.path.join(exps_folders[i][l], f'metrics_full_df_r{subject_name}.csv'))
                metrics_df[METRICS_NAMES[k]].append(subject_metric_df[metric_column].dropna().values.mean())

            # concatenate all values from dataframes, and compute mean and std
            metrics_df[METRICS_NAMES[k]] = [np.array(metrics_df[METRICS_NAMES[k]]).mean(), np.array(metrics_df[METRICS_NAMES[k]]).std()]
        
        # append subject data
        subjects_df[subject_name] = metrics_df
    
    # append task data
    tasks_dfs[task_name] = subjects_df

# plot graph
y_start = 0.1
step_size = 1
eps = 1.5e-1
x = step_size - 1.5*eps
x_text = step_size
y_text_step = 0.1
for j, subject_name in enumerate(SUBJECTS_NAMES):

    # plot for each task and advance
    for i, task_name in enumerate(TASKS_NAMES):
        
        for k, metric_name in enumerate(METRICS_NAMES):
            # draw shape and write text
            plt.scatter(x, tasks_dfs[task_name][subject_name][metric_name][0], marker=METRICS_SYMBOLS[k], s=80, color=COLORS[2*i+k])
            text_loc_y = y_start + 0.01 + 3 * y_text_step - (y_text_step * (2*i+k))
            plt.text(x_text - 2.5*eps, text_loc_y, str(round(tasks_dfs[task_name][subject_name][metric_name][0], 3)), color=COLORS[2*i+k], fontsize=FONT_SIZE-2)

            # draw std line
            lower_bound = max(0, tasks_dfs[task_name][subject_name][metric_name][0] - tasks_dfs[task_name][subject_name][metric_name][1])
            upper_bound = min(1, tasks_dfs[task_name][subject_name][metric_name][0] + tasks_dfs[task_name][subject_name][metric_name][1])
            plt.plot([x]*2, [lower_bound, upper_bound], color=COLORS[2*i+k])

            x += eps
        
    x += (step_size - 4*eps)
    x_text += step_size

# prepare customized legend
legend_symbols_colors = ['darkgrey', 'dimgrey']
legend_lines = [Line2D([0], [0], color=COLORS[0], label=TASKS_NAMES_SHORT[0], linewidth=3, linestyle='-'),Line2D([0], [0], color=COLORS[2], label=TASKS_NAMES_SHORT[1], linewidth=3, linestyle='-')]
legend_symbols = [Line2D([], [], color=z, marker=x, linestyle='None', markersize=8, label=y) for x,y,z in zip(METRICS_SYMBOLS, METRICS_NAMES,legend_symbols_colors)]
metrics_legend = plt.legend(handles=legend_lines+legend_symbols, loc='lower right', fontsize=FONT_SIZE-3)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(range(1, len(SUBJECTS_NAMES)+1), range(1, len(SUBJECTS_NAMES)+1), fontsize=FONT_SIZE-2)
plt.yticks(np.arange(0,1.25, step=0.25), fontsize=FONT_SIZE)
plt.xlim([0.6, 15.0])
plt.ylim([y_start, 1.0])
plt.xlabel('Subject', fontsize=FONT_SIZE-3)

# save plot
plt.grid(True, which='major')
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_subjects_cigraph_combined.{}".format(GRAPH_FORMAT)))