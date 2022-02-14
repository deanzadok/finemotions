import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

METHODS_NAMES = ['1 image (SF)','2 images (MF)', '4 images (MF)', '6 images (MF)', '8 images (MF)']
NUM_EPOCHS = 20
GRAPH_FORMAT = 'pdf'
GRAPH_TITLE = 'Number of Images'

methods_folders = ['/mnt/walkure_public/deanz/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_01_st0.8_kf4',
                   '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_2imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_4imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_6imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/deanz/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_03_st0.8_sequence_kf4']

bce_losses = np.zeros((len(METHODS_NAMES),NUM_EPOCHS))
for i, method_dir in enumerate(methods_folders):
    bce_losses[i, :] = pd.read_csv(os.path.join(method_dir, 'metric.csv'))['test_dev_loss'].iloc[:NUM_EPOCHS].values

# plot graph
for i in range(len(METHODS_NAMES)):

    plt.plot(range(1, NUM_EPOCHS+1), bce_losses[i, :], label=METHODS_NAMES[i])
plt.legend()
plt.xticks(range(1, NUM_EPOCHS+1, 2))
plt.xlim([1, NUM_EPOCHS])
plt.xlabel('Epoch')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel(r"$\mathcal{L}_{BCE}$")
plt.grid(True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("imgcomp_graph.{}".format(GRAPH_FORMAT)))