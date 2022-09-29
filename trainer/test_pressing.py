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
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/test_mfm_unet_aida_r00_gen_us2conf2multimidi.json', type=str)
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
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.system.gpu)

    # check if output folder exists
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # load train and test datasets
    data_mng = DataManagement(cfg=cfg)

    # create model and load weights
    model = initiate_model(cfg=cfg)
    model.load_weights(cfg.model.weights)

    print('Start testing...')
    metrics_np = np.zeros((len(data_mng.test_df), 4))
    if cfg.data.sequence:
        metrics_classes_np = np.zeros((len(data_mng.test_df), cfg.data.dev_classes, 4))
    predictions_np = np.zeros((len(data_mng.test_df), cfg.data.dev_classes))
    labels_np = np.zeros((len(data_mng.test_df), cfg.data.dev_classes))
    images_np = None
    if cfg.use_imgs:
        images_np = np.zeros((len(data_mng.test_df), cfg.data.res, cfg.data.res))

    print('GPU: {}, Weights: {}'.format(cfg.system.gpu,cfg.model.weights))

    # iterate test session samples
    for i, batch in enumerate(data_mng.test_gen):

        # get predictions
        if cfg.mode == 'us2confNmultimidi':
            _, images, _, dev_y = batch
            predictions = predict(images=images, dev_x=None, dev_y=dev_y, mode=cfg.mode)
        elif cfg.mode == 'us2midi':
            images, dev_x, dev_y = batch
            predictions = predict(images=images, dev_x=dev_x, dev_y=dev_y, mode=cfg.mode)
        elif cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey':
            images, dev_y = batch
            predictions = predict(images=images, dev_x=None, dev_y=dev_y, mode=cfg.mode)
        elif cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
            _, images, _, dev_y = batch
            predictions = predict(images=images, dev_x=None, dev_y=dev_y, mode=cfg.mode)
        elif cfg.mode == 'multimidi2multimidi':
            dev_y = batch
            predictions = predict(images=None, dev_x=None, dev_y=dev_y, mode=cfg.mode)
        else: # cfg.mode == 'midi2midi'
            dev_x, dev_y = batch
            predictions = predict(images=None, dev_x=dev_x, dev_y=dev_y, mode=cfg.mode)

        # compute confusion matrix stats and store them
        metrics_np[i,:] = compute_dev_metrics(dev_y, tf.sigmoid(predictions), sequence=cfg.data.sequence)

        # store predictions and inputs
        if cfg.data.sequence:
            # compute confusion matrix for each class
            metrics_classes_np[i,:,:] = compute_dev_metrics_per_class(dev_y, tf.sigmoid(predictions), sequence=cfg.data.sequence)

            predictions_np[i,:] = tf.sigmoid(predictions[0,-1,:]).numpy()
            labels_np[i,:] = dev_y[0,-1,:].numpy()
        else:
            predictions_np[i,:] = tf.sigmoid(predictions[0,:]).numpy()
            labels_np[i,:] = dev_y[0,:].numpy()
        if cfg.use_imgs:
            images_np[i,:] = images[0,:,:,-1].numpy()

    # printing
    acc_values = metrics_np[:,0][~np.isnan(metrics_np[:,0])]
    rec_values = metrics_np[:,1][~np.isnan(metrics_np[:,1])]
    pre_values = metrics_np[:,2][~np.isnan(metrics_np[:,2])]
    f1_values = metrics_np[:,3][~np.isnan(metrics_np[:,3])]
    print('Accuracy mean: {:.5f}, std: {:.5f}.'.format(acc_values.mean(), acc_values.std()), end=" ")
    print('Recall mean: {:.5f}, std: {:.5f}.'.format(rec_values.mean(), rec_values.std()), end=" ")
    print('Precision mean: {:.5f}, std: {:.5f}.'.format(pre_values.mean(), pre_values.std()), end=" ")
    print('F1 mean: {:.5f}, std: {:.5f}.'.format(f1_values.mean(), f1_values.std()))

    # save raw metrics
    metrics_df = pd.DataFrame(data=metrics_np, columns=['acc','rec','pre','f1'])
    if cfg.output_name != "":
        metrics_df.to_csv(os.path.join(os.path.dirname(cfg.model.weights), f'metrics_full_df_{cfg.output_name}.csv'), index=False)
    else:
        metrics_df.to_csv(os.path.join(os.path.dirname(cfg.model.weights), 'metrics_full_df.csv'), index=False)

    # create session visualization
    #create_playing_visualization(images=images_np, preds=predictions_np, dev_y=labels_np)

    # store raw predictions
    # store_raw_predictions_and_images(cfg=cfg, images=images_np, labels=labels_np, predictions=predictions_np)

    # store classes metrics dataframe
    #if cfg.data.sequence:
    #   store_metrics_per_class(metrics_classes_np=metrics_classes_np, data_dir=os.path.dirname(cfg.model.weights))