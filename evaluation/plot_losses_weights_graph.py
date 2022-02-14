from cProfile import label
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

NUM_EPOCHS = 20
GRAPH_FORMAT = 'pdf'

losses_weights = [-1, 0, 1, 2, 3, 4, 5, 6]
methods_folders = ['/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_10_st0.8_sequence_reslayer_retrained_0qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_09_st0.8_sequence_reslayer_retrained',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_11_st0.8_sequence_reslayer_retrained_2qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_12_st0.8_sequence_reslayer_retrained_4qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_13_st0.8_sequence_reslayer_retrained_8qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_14_st0.8_sequence_reslayer_retrained_16qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_15_st0.8_sequence_reslayer_retrained_32qloss',
                   '/mnt/walkure_public/deanz/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_16_st0.8_sequence_reslayer_retrained_64qloss']


methods_losses = {}
for i, method_dir in enumerate(methods_folders):
    # append all k-fold xperiments into one dataframe
    metrics_df = pd.read_csv(os.path.join(method_dir, 'metric.csv'))

    methods_losses[losses_weights[i]] = [metrics_df.test_dev_loss.iloc[NUM_EPOCHS-1], metrics_df.test_c_loss.iloc[NUM_EPOCHS-1]]

# plot graph
for key, val in methods_losses.items():
    if key >= 0:
        plt.scatter(val[1], val[0], marker='d', s=80, label=rf"$2^{key}$")


plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"$\mathcal{L}_{MSE}$")
plt.ylabel(r"$\mathcal{L}_{BCE}$")
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_losses_weights.{}".format(GRAPH_FORMAT)))
