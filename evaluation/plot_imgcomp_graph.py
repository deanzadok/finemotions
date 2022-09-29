import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (5,4)
METHODS_NAMES = ['1 image (SF)','2 images (MF)', '4 images (MF)', '6 images (MF)', '8 images (MF)']
NUM_EPOCHS = 20
GRAPH_FORMAT = 'pdf'
GRAPH_TITLE = 'Number of Images'

# define experiments folders
methods_folders = ['/mnt/walkure_public/username/models/deepnet/us2multimidi_all/deepnetunet_224res_1imgs_calib_all_multiplaying_01_st0.8_kf4',
                   '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_2imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_4imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_6imgs_calib_all_multiplaying_01_st0.8_sequence_kf4', 
                   '/mnt/walkure_public/username/models/mfm/us2multimidi_all/mfmunet_224res_8imgs_calib_all_multiplaying_03_st0.8_sequence_kf4']

# load bce losses for each one of the models
bce_losses = np.zeros((len(METHODS_NAMES),NUM_EPOCHS))
for i, method_dir in enumerate(methods_folders):
    bce_losses[i, :] = pd.read_csv(os.path.join(method_dir, 'metric.csv'))['test_dev_loss'].iloc[:NUM_EPOCHS].values

# plot graph
for i in range(len(METHODS_NAMES)):
    plt.plot(range(1, NUM_EPOCHS+1), bce_losses[i, :], label=METHODS_NAMES[i])

# prepare legend
plt.legend(fontsize=FONT_SIZE-3)

# prepare axes
plt.xticks(range(1, NUM_EPOCHS+1, 2), fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlim([1, NUM_EPOCHS])
plt.xlabel('Epoch')
plt.ylabel(r"$\mathcal{L}_{BCE}$")

# save plot
plt.grid(True)
plt.tight_layout(pad=0.7)
plt.savefig(os.path.join("imgcomp_graph.{}".format(GRAPH_FORMAT)))