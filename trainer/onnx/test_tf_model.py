from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from data.load import DataManagement
from trainer.utils import ConfigManager, wait_for_gpu, initiate_model, compute_dev_metrics, compute_dev_metrics_per_class, create_playing_visualization, store_metrics_per_class, store_raw_predictions_and_images

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/test_mfm_aida_all_us2conf2multimidi.json', type=str)
args = parser.parse_args()

# tf function to predict
@tf.function
def predict(images, dev_x, dev_y, mode):

    # get predictions
    if mode == 'us2confNmultimidi':
        _, pred_midis, _, _ = model(images)
    elif mode == 'midi2midi':
        pred_midis, _, _ = model([dev_y, dev_x])
    elif mode == 'us2multimidi' or mode == 'us2multikey':
        #pred_midis, _, _ = model(images)
        pred_midis = model(images)
    elif mode == 'us2conf2multimidi' or mode == 'us2conf2multikey':
        #pred_midis, _, _ = model(images)
        _, pred_midis = model(images)
    elif mode == 'multimidi2multimidi':
        pred_midis, _, _ = model(dev_y)
    else: # mode == 'us2midi'
        pred_midis, _, _ = model([images, dev_x])

    return pred_midis
    
if __name__ == "__main__":

    # load config file
    cfg = ConfigManager(json_name=args.json)

    # list visible devices and use allow growth - updated for TF 2.7 (CUDA 11 + CUDNN 8.2)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices([gpus[cfg.system.gpu]], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[cfg.system.gpu], True)

    # check if output folder exists
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # load train and test datasets
    data_mng = DataManagement(cfg=cfg)

    # create model and load weights
    model = initiate_model(cfg=cfg)
    model.load_weights(cfg.model.weights)

    # perform single prediction
    _, images, _, dev_y = next(iter(data_mng.test_gen))
    predictions = predict(images=images, dev_x=None, dev_y=dev_y, mode=cfg.mode)

    print(predictions[-1,-1,:].numpy().tolist())    