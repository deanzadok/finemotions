import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (5,4)
METHODS_NAMES = ['SF', 'MF', 'CBMF']
NUM_EPOCHS = 20
NUM_EXPS = 5
GRAPH_FORMAT = 'pdf'
# GRAPH_TITLE = 'Piano Playing'
# GRAPH_FILE = 'piano_playing'
GRAPH_TITLE = 'Keyboard Typing'
GRAPH_FILE = 'keyboard_typing'

# define folders of experiments
if GRAPH_FILE == 'piano_playing':
    methods_folders = [['/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_01_st0.8_kf4',
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
                        '/mnt/walkure_public/username/models/mfm/us2conf2multimidi_all/mfmunet_224res_8imgs_calib_all_multityping_17_st0.8_sequence_reslayer_retrained_mp_4qloss_kf3']]
else:
    methods_folders = [['/mnt/walkure_public/username/models/deepnet/us2multikey_all/deepnetunet_224res_1imgs_calib_all_multityping_01_st0.8_kf4',
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
                        '/mnt/walkure_public/username/models/mfm/us2conf2multikey_all/mfmunet_224res_8imgs_calib_all_multityping_27_st0.8_sequence_reslayer_retrained_mt_4qloss_kf3']]

# read test device losses
bce_losses = np.zeros((len(METHODS_NAMES),NUM_EXPS,NUM_EPOCHS))
for i, method_exps in enumerate(methods_folders):
    for j, exp_dir in enumerate(method_exps):
        bce_losses[i, j, :] = pd.read_csv(os.path.join(exp_dir, 'metric.csv'))['test_dev_loss'].iloc[:NUM_EPOCHS].values

# plot graph
bce_losses_features = np.zeros((len(METHODS_NAMES),3,NUM_EPOCHS))
for i in range(len(METHODS_NAMES)):
    bce_losses_features[i, 0, :] = bce_losses[i,:,:].mean(axis=0)
    bce_losses_features[i, 1, :] = bce_losses[i,:,:].min(axis=0)
    bce_losses_features[i, 2, :] = bce_losses[i,:,:].max(axis=0)

    plt.plot(range(1, NUM_EPOCHS+1), bce_losses_features[i, 0, :], label=METHODS_NAMES[i])
    plt.fill_between(range(1, NUM_EPOCHS+1), bce_losses_features[i, 0, :] + bce_losses[i,:,:].std(axis=0), bce_losses_features[i, 0, :] - bce_losses[i,:,:].std(axis=0), alpha=0.5)

# print legend
plt.legend(loc='upper right', fontsize=FONT_SIZE-2)

# prepare axes
plt.xticks(range(1, NUM_EPOCHS+1, 2), fontsize=FONT_SIZE)
#plt.yticks(np.arange(0.05, 0.45, step=0.05), fontsize=FONT_SIZE)
plt.xlim([1, NUM_EPOCHS])
#plt.ylim([0, 0.45])
plt.xlabel('Epoch')
plt.ylabel(r"$\mathcal{L}_{BCE}$")

# save plot
plt.grid(True)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join("bce_graph_{}.{}".format(GRAPH_FILE, GRAPH_FORMAT)))