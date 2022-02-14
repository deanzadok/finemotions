from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from tensorflow.python.framework.ops import prepend_name_scope
sys.path.append('.')
sys.path.append('..')
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from data.load import DataManagement
from trainer.losses import ForwardKinematicsError, ConfigurationDynamicsError, ForwardDynamicsError 
from trainer.utils import ConfigManager, wait_for_gpu, initiate_model, set_random_seed
import time
import subprocess as sp

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/train_mfm_unet_aida_all_us2multikey_debug.json', type=str)
args = parser.parse_args()

# tf function to train
@tf.function
def train(timestamps, images, conf_y, dev_x, dev_y, cfg):

    with tf.GradientTape() as tape:

        # get predictions
        if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
            pred_confs, pred_devs = model(images)
        elif cfg.mode == 'us2conf':
            pred_confs = model(images)
        else:
            pred_devs = model(images)

        # compute task losses
        if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
            q_loss = mse(conf_y, pred_confs)
            dev_loss = bce(dev_y, pred_devs)
            loss = 4*q_loss + dev_loss
        elif cfg.mode == 'us2conf':
            q_loss = mse(conf_y, pred_confs)
            loss = q_loss
        else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
            dev_loss = bce(dev_y, pred_devs)
            loss = dev_loss

        # add FK loss
        if cfg.data.use_fk:
            fk_loss = fke(conf_y, pred_confs)
            loss += cfg.data.fk * fk_loss
            train_fk_loss(fk_loss)

        # add CD loss
        if cfg.data.use_cd or cfg.data.use_fd:
            cde.set_time(timestamps)
            cd_loss = cde(conf_y, pred_confs)
        if cfg.data.use_cd:
            loss += cfg.data.cd * cd_loss
            train_cd_loss(cd_loss)

        # add FD loss
        if cfg.data.use_fd:
            fde.set_time_and_grads(timestamps, cde.y_true_grad, cde.y_pred_grad)
            fd_loss = fde(conf_y, pred_confs)
            loss += cfg.data.fd * fd_loss
            train_fd_loss(fd_loss)

    # perform optimization step
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
        train_c_loss(q_loss)
        train_dev_loss(dev_loss)        
    elif cfg.mode == 'us2conf':
        train_c_loss(q_loss)
    else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
        train_dev_loss(dev_loss)

# tf function to test
@tf.function
def test(timestamps, images, conf_y, dev_x, dev_y, cfg):
    
    # get predictions
    if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
        pred_confs, pred_devs = model(images)
    elif cfg.mode == 'us2conf':
        pred_confs = model(images)
    else:
        pred_devs = model(images)

    # compute task losses
    if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
        q_loss = mse(conf_y, pred_confs)
        dev_loss = bce(dev_y, pred_devs)
    elif cfg.mode == 'us2conf':
        q_loss = mse(conf_y, pred_confs)
    else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
        dev_loss = bce(dev_y, pred_devs)
            
    # log tasks losses
    if cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
        test_c_loss(q_loss)
        test_dev_loss(dev_loss)
    elif cfg.mode == 'us2conf':
        test_c_loss(q_loss)
    else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
        test_dev_loss(dev_loss)

    # log FK loss
    if cfg.data.use_fk:
        fk_loss = fke(conf_y, pred_confs)
        test_fk_loss(fk_loss)

    # log CD loss
    if cfg.data.use_cd or cfg.data.use_fd:
        cde.set_time(timestamps)
        cd_loss = cde(conf_y, pred_confs)
    if cfg.data.use_cd:
        test_cd_loss(tf.cast(cd_loss, q_loss.dtype))

    # log FD loss
    if cfg.data.use_fd:
        fde.set_time_and_grads(timestamps, cde.y_true_grad, cde.y_pred_grad)
        fd_loss = fde(conf_y, pred_confs)
        test_fd_loss(tf.cast(fd_loss, q_loss.dtype))

if __name__ == "__main__":

    # load config file
    cfg = ConfigManager(json_name=args.json, retrain=True)

    # set random seed (do nothing for no random seed)
    set_random_seed(cfg)

    # list visible devices and use allow growth - updated for TF 2.7 (CUDA 11 + CUDNN 8.2)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices([gpus[cfg.system.gpu]], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[cfg.system.gpu], True)

    # check if output folder exists
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # initiate csv of training if asked
    if cfg.store_csv:
        metric_df_columns = []
        if cfg.use_conf:
            metric_df_columns += ['train_c_loss','test_c_loss']
        if cfg.data.use_fk:
            metric_df_columns += ['train_fk_loss','test_fk_loss']
        if cfg.data.use_fd:
            metric_df_columns += ['train_fd_loss','test_fd_loss']
        if cfg.data.use_cd:
            metric_df_columns += ['train_cd_loss','test_cd_loss']
        if cfg.use_dev:
            metric_df_columns += ['train_dev_loss','test_dev_loss']
        metric_df = pd.DataFrame(columns=metric_df_columns)

    # load train and test datasets
    data_mng = DataManagement(cfg=cfg)

    # wait for gpu if asked
    wait_for_gpu(gpu=str(cfg.system.gpu), memory_req=cfg.system.memory_req)

    # create model, loss and optimizer
    model = initiate_model(cfg=cfg)
    if cfg.use_conf:
        mse = tf.keras.losses.MeanSquaredError()
    if cfg.use_dev:
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    if cfg.data.use_fk:
        fke = ForwardKinematicsError(cfg=cfg, arm_lengths=data_mng.arm_lengths)
    if cfg.data.use_cd or cfg.data.use_fd:
        cde = ConfigurationDynamicsError(cfg=cfg)
    if cfg.data.use_fd:
        fde = ForwardDynamicsError(cfg=cfg, arm_lengths=data_mng.arm_lengths)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)

    # load weights
    if cfg.model.weights != "":
        model.load_weights(cfg.model.weights)

    # define metrics
    if cfg.use_conf:
        train_c_loss = tf.keras.metrics.Mean(name='train_c_loss')
        test_c_loss = tf.keras.metrics.Mean(name='test_c_loss')
    if cfg.data.use_fk:
        train_fk_loss = tf.keras.metrics.Mean(name='train_fk_loss')
        test_fk_loss = tf.keras.metrics.Mean(name='test_fk_loss')
    if cfg.data.use_cd:
        train_cd_loss = tf.keras.metrics.Mean(name='train_cd_loss')
        test_cd_loss = tf.keras.metrics.Mean(name='test_cd_loss')
    if cfg.data.use_fd:
        train_fd_loss = tf.keras.metrics.Mean(name='train_fd_loss')
        test_fd_loss = tf.keras.metrics.Mean(name='test_fd_loss')
    if cfg.use_dev:
        train_dev_loss = tf.keras.metrics.Mean(name='train_dev_loss')
        test_dev_loss = tf.keras.metrics.Mean(name='test_dev_loss')
    metrics_writer = tf.summary.create_file_writer(cfg.output_dir)

    # train
    train_counter = 0
    test_counter = 0
    print('Start training...')
    for epoch in range(cfg.training.epochs):
        
        for batch in data_mng.train_gen:
            if cfg.mode == 'us2conf':
                _, images, conf_y = batch
                train(timestamps=None, images=images, conf_y=conf_y, dev_x=None, dev_y=None, cfg=cfg)
            elif cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
                timestamps, images, conf_y, dev_y = batch
                train(timestamps=timestamps, images=images, conf_y=conf_y, dev_x=None, dev_y=dev_y, cfg=cfg)
            else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
                images, dev_y = batch
                train(timestamps=None, images=images, conf_y=None, dev_x=None, dev_y=dev_y, cfg=cfg)

            train_counter += 1
            with metrics_writer.as_default():
                if cfg.use_conf:
                    tf.summary.scalar('Train C loss', train_c_loss.result(), step=test_counter)
                if cfg.data.use_fk:
                    tf.summary.scalar('Train FK loss', train_fk_loss.result(), step=test_counter)
                if cfg.data.use_cd:
                    tf.summary.scalar('Train CD loss', train_cd_loss.result(), step=test_counter)
                if cfg.data.use_fd:
                    tf.summary.scalar('Train FD loss', train_fd_loss.result(), step=test_counter)
                if cfg.use_dev:
                    tf.summary.scalar('Train Dev loss', train_dev_loss.result(), step=test_counter)

        for test_batch in data_mng.test_gen:
            if cfg.mode == 'us2conf':
                _, test_images, test_conf_y = test_batch
                test(timestamps=None, images=test_images, conf_y=test_conf_y, dev_x=None, dev_y=None, cfg=cfg)
            elif cfg.mode == 'us2conf2multimidi' or cfg.mode == 'us2conf2multikey':
                test_timestamps, test_images, test_conf_y, test_dev_y = test_batch
                test(timestamps=test_timestamps, images=test_images, conf_y=test_conf_y, dev_x=None, dev_y=test_dev_y, cfg=cfg)
            else: # cfg.mode == 'us2multimidi' or cfg.mode == 'us2multikey'
                test_images, test_dev_y = test_batch
                test(timestamps=None, images=test_images, conf_y=None, dev_x=None, dev_y=test_dev_y, cfg=cfg)

            test_counter += 1
            with metrics_writer.as_default():
                if cfg.use_conf:
                    tf.summary.scalar('Test C loss', test_c_loss.result(), step=test_counter)
                if cfg.data.use_fk:
                    tf.summary.scalar('Test FK loss', test_fk_loss.result(), step=test_counter)
                if cfg.data.use_cd:
                    tf.summary.scalar('Test CD loss', test_cd_loss.result(), step=test_counter)
                if cfg.data.use_fd:
                    tf.summary.scalar('Test FD loss', test_fd_loss.result(), step=test_counter)
                if cfg.use_dev:
                    tf.summary.scalar('Test Dev loss', test_dev_loss.result(), step=test_counter)


        if cfg.store_csv:
            row_dict = {}
            if cfg.use_conf:
                row_dict['train_c_loss'] = train_c_loss.result().numpy()
                row_dict['test_c_loss'] = test_c_loss.result().numpy()
            if cfg.data.use_fk:
                row_dict['train_fk_loss'] = train_fk_loss.result().numpy()
                row_dict['test_fk_loss'] = test_fk_loss.result().numpy()
            if cfg.data.use_fd:
                row_dict['train_fd_loss'] = train_fd_loss.result().numpy()
                row_dict['test_fd_loss'] = test_fd_loss.result().numpy()
            if cfg.data.use_cd:
                row_dict['train_cd_loss'] = train_cd_loss.result().numpy()
                row_dict['test_cd_loss'] = test_cd_loss.result().numpy()
            if cfg.use_dev:
                row_dict['train_dev_loss'] = train_dev_loss.result().numpy()
                row_dict['test_dev_loss'] = test_dev_loss.result().numpy()
            metric_df = metric_df.append(row_dict, ignore_index=True)
            metric_df.to_csv(path_or_buf=os.path.join(cfg.output_dir, 'metric.csv'), index=False)

        # printing
        print('Epoch {},'.format(epoch+1), end=" ")
        if cfg.use_conf:
            print('C L: {:.5f}, T C L: {:.5f}.'.format(train_c_loss.result(), test_c_loss.result()), end=" ")
        if cfg.data.use_fk:
            print('FK L: {:.5f}, T FK L: {:.5f}.'.format(train_fk_loss.result(), test_fk_loss.result()), end=" ")
        if cfg.data.use_cd:
            print('CD L: {:.5f}, T CD L: {:.5f}.'.format(train_cd_loss.result(), test_cd_loss.result()), end=" ")
        if cfg.data.use_fd:
            print('FD L: {:.5f}, T FD L: {:.5f}.'.format(train_fd_loss.result(), test_fd_loss.result()), end=" ")
        if cfg.use_dev:
            print('Dev L: {:.5f}, T Dev L: {:.5f}.'.format(train_dev_loss.result(), test_dev_loss.result()), end=" ")

        print(" ")
        if (epoch+1) % 10 == 0 or epoch == 0:
            print('GPU: {}, Experiment: {}'.format(cfg.system.gpu,cfg.output_dir))
        
        # save model
        if (epoch+1) % cfg.training.cp_interval == 0 and epoch > 0:
            print('Saving weights to {}'.format(cfg.output_dir))
            model.save_weights(os.path.join(cfg.output_dir, "model{}.ckpt".format(epoch+1)))