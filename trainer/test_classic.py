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
from trainer.utils import ConfigManager, wait_for_gpu, initiate_model, initiate_classifier, filter_batch, compute_dev_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/classic/test_classic_aida_all_us2multikey.json', type=str)
args = parser.parse_args()

# tf function to predict
@tf.function
def predict(imgs, cfg):

    # get predictions
    preds = model(imgs)

    return preds
    
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

    print('Start testing...')
    pred_features_train = []
    pred_features_test = []
    #predictions_np = []
    labels_train = []
    labels_test = []

    print('GPU: {}, Weights: {}'.format(cfg.system.gpu,cfg.model.weights))

    # iterate train session samples
    for i, batch in enumerate(data_mng.train_gen):

        # get predictions
        timestamps, imgs, _, _, y_dev = filter_batch(batch=batch, cfg=cfg)
        predictions = predict(imgs=imgs, cfg=cfg)
        pred_features_train.append(predictions.numpy())
        labels_train.append(y_dev[:,-1,:].numpy())
        if i % 10 == 0:
            print('downsample-train {}/{}'.format(i*timestamps.shape[0], len(data_mng.train_df)))

    # iterate test session samples
    for i, batch in enumerate(data_mng.test_gen):

        # get predictions
        timestamps, imgs, _, _, y_dev = filter_batch(batch=batch, cfg=cfg)
        predictions = predict(imgs=imgs, cfg=cfg)
        pred_features_test.append(predictions.numpy())
        labels_test.append(y_dev[:,-1,:].numpy())
        if i % 10 == 0:
            print('downsample-test {}/{}'.format(i*timestamps.shape[0], len(data_mng.test_df)))

    # concatenate data
    pred_features_train = np.concatenate(pred_features_train, axis=0)
    pred_features_test = np.concatenate(pred_features_test, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    labels_test = np.concatenate(labels_test, axis=0)

    if cfg.model.type == 'perfor':
        pred_features_train_pf = np.zeros_like(pred_features_train[:,:,:2])
        x = range(20)
        for i in range(pred_features_train.shape[0]):
            for j in range(pred_features_train.shape[1]):
                pred_features_train_pf[i,j] = np.polyfit(x, pred_features_train[i,j], 1)
            if i % 1000 == 0:
                print('polyfit-train {}/{}'.format(i, pred_features_train.shape[0]))
        
        pred_features_test_pf = np.zeros_like(pred_features_test[:,:,:2])
        x = range(20)
        for i in range(pred_features_test.shape[0]):
            for j in range(pred_features_test.shape[1]):
                pred_features_test_pf[i,j] = np.polyfit(x, pred_features_test[i,j], 1)
            if i % 1000 == 0:
                print('polyfit-test {}/{}'.format(i, pred_features_test.shape[0]))
        
        pred_features_train = pred_features_train_pf.reshape(pred_features_train_pf.shape[0],-1)
        pred_features_test = pred_features_test_pf.reshape(pred_features_test_pf.shape[0],-1)
    else:
        # normalize
        train_mean = pred_features_train.mean(axis=0)
        train_std = pred_features_train.std(axis=0)
        pred_features_train = (pred_features_train - train_mean) / train_std
        pred_features_test = (pred_features_test - train_mean) / train_std

    # prepare classes for classification
    powers_np = 2**np.arange(cfg.data.dev_classes)[::-1]
    labels_train_dec = np.zeros(labels_train.shape[0])
    for i in range(labels_train.shape[0]):
        labels_train_dec[i] = labels_train[i].dot(powers_np)
    labels_test_dec = np.zeros(labels_test.shape[0])
    for i in range(labels_test.shape[0]):
        labels_test_dec[i] = labels_test[i].dot(powers_np)

    # classify
    print("Started classification...")
    cls = initiate_classifier(cfg=cfg)
    cls.fit(pred_features_train, labels_train_dec)
    labels_pred_dec = cls.predict(pred_features_test)

    # return to class structure
    labels_pred = np.zeros((labels_pred_dec.shape[0], cfg.data.dev_classes), dtype=np.int32)
    for i in range(labels_test.shape[0]):
        bin_vals = [int(x) for x in list(bin(int(labels_pred_dec[i]))[2:])]
        bin_vals = np.array([0] * (cfg.data.dev_classes - len(bin_vals)) + bin_vals)
        labels_pred[i] = bin_vals

    # compute metrics
    labels_test_tf = tf.convert_to_tensor(labels_test)
    labels_pred_tf = tf.convert_to_tensor(labels_pred, dtype=tf.float32)
    acc_val, rec_val, pre_val, f1_val = compute_dev_metrics(dev_y=labels_test_tf, predictions=labels_pred_tf, sequence=False)
    print('Accuracy: {:.5f}.'.format(acc_val.numpy()), end=" ")
    print('Recall: {:.5f}.'.format(rec_val.numpy()), end=" ")
    print('Precision: {:.5f}.'.format(pre_val.numpy()), end=" ")
    print('F1: {:.5f}.'.format(f1_val.numpy()))

    # save metrics
    metrics_df = pd.DataFrame(columns=['acc','rec','pre','f1'])
    metrics_row = {'acc':acc_val.numpy(), 'rec':rec_val.numpy(), 'pre':pre_val.numpy(), 'f1':f1_val.numpy()}
    metrics_df = metrics_df.append(metrics_row, ignore_index=True)
    if cfg.output_name != "":
        metrics_df.to_csv(os.path.join(cfg.output_dir, f'metrics_full_df_mean_{cfg.output_name}.csv'), index=False)
    else:
        metrics_df.to_csv(os.path.join(cfg.output_dir, 'metrics_full_df_mean.csv'), index=False)