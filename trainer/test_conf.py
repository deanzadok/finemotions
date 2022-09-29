from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data.load import DataManagement
from trainer.utils import ConfigManager, wait_for_gpu, initiate_model, filter_batch, compute_conf_metrics, create_confs_graph, store_conf_metrics, store_raw_predictions_and_images
from trainer.losses import ForwardKinematicsError, ConfigurationDynamicsError, ForwardDynamicsError

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/test_mfm_aida_all_us2conf.json', type=str)
args = parser.parse_args()

# tf function to predict
@tf.function
def predict(imgs, cfg):

    # get predictions
    if cfg.mode == 'us2conf':
        pred_confs = model(imgs)
    elif cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
        pred_confs, _ = model(imgs)
    elif cfg.mode == 'us2usNconf':
        _, pred_confs = model(imgs)
    else: # cfg.mode == 'us2confNmultimidi' or cfg.mode == 'us2confNmultikey':
        pred_confs, _, _, _ = model(imgs)

    return pred_confs
    
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

    # create loss functions
    ike = tf.keras.losses.MeanSquaredError()
    fke = ForwardKinematicsError(cfg=cfg, arm_lengths=data_mng.arm_lengths)
    ide = ConfigurationDynamicsError(cfg=cfg)
    fde = ForwardDynamicsError(cfg=cfg, arm_lengths=data_mng.arm_lengths)

    print('Start testing...')
    metrics_np = np.zeros((len(data_mng.test_df), 4))
    predictions_np = np.zeros((len(data_mng.test_df), model.num_outputs))
    labels_np = np.zeros((len(data_mng.test_df), model.num_outputs))
    images_np = None
    if cfg.use_imgs:
        images_np = np.zeros((len(data_mng.test_df), cfg.data.res, cfg.data.res))
    ik_errors, fk_errors = [], []

    # iterate test session samples
    for i, batch in enumerate(data_mng.test_gen):

        # get predictions
        timestamps, imgs, _, _, y_conf, _ = filter_batch(batch=batch, cfg=cfg)
        predictions = predict(imgs=imgs, cfg=cfg)

        # compute confusion matrix stats and store them
        ik_val, fk_val, id_val, fd_val, ik_errs, fk_errs = compute_conf_metrics(conf_y=y_conf, preds=predictions, timestamps=timestamps, ike=ike, fke=fke, ide=ide, fde=fde, is_sequence=cfg.data.sequence)
        ik_errors.append(ik_errs)
        if cfg.data.sequence:
            fk_errors.append(fk_errs)

        # store predictions and inputs
        metrics_np[i] = [ik_val, fk_val, id_val, fd_val]
        if cfg.data.sequence:
            predictions_np[i,:] = predictions[0,-1,:].numpy()
            labels_np[i,:] = y_conf[0,-1,:].numpy()
        else:
            predictions_np[i,:] = predictions[0,:].numpy()
            labels_np[i,:] = y_conf[0,:].numpy()

        if cfg.use_imgs:
            images_np[i,:] = imgs[0,:,:,-1].numpy()

    # printing
    ik_values = metrics_np[:,0][~np.isnan(metrics_np[:,0])]
    print('IK mean: {:.5f}, std: {:.5f}.'.format(ik_values.mean(), ik_values.std()), end=" ")

    ik_errors = tf.concat(ik_errors, axis=0).numpy()

    if cfg.data.sequence:
        fk_values = metrics_np[:,1][~np.isnan(metrics_np[:,1])]
        id_values = metrics_np[:,2][~np.isnan(metrics_np[:,2])]
        fd_values = metrics_np[:,3][~np.isnan(metrics_np[:,3])]
        print('FK mean: {:.5f}, std: {:.5f}.'.format(fk_values.mean(), fk_values.std()), end=" ")
        print('ID mean: {:.5f}, std: {:.5f}.'.format(id_values.mean(), id_values.std()), end=" ")
        print('FD mean: {:.5f}, std: {:.5f}.'.format(fd_values.mean(), fd_values.std()))

        fk_errors = tf.concat(fk_errors, axis=0).numpy()

    # store confs metrics for printing
    #store_conf_metrics(data_dir=os.path.dirname(cfg.model.weights), ik_errors=ik_errors, fk_errors=fk_errors, ik_labels=data_mng.labels_names, fk_labels=['finger4', 'finger3', 'finger2', 'finger1', 'thumb'])

    # create session visualization
    create_confs_graph(preds_np=predictions_np, labels_np=labels_np, labels_names=data_mng.labels_names_stacked)

    # store raw predictions
    #store_raw_predictions_and_images(cfg=cfg, images=images_np, labels=labels_np, predictions=predictions_np, output_dir=os.path.dirname(cfg.model.weights), store_imgs=False, classes=data_mng.labels_names, arm_lengths=data_mng.arm_lengths, palm_vectors=data_mng.palm_vectors)