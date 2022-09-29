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
from trainer.models.mf import MultiFrameModelRecOnly
import tf2onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/test_mfm_aida_all_us2conf2multimidi.json', type=str)
args = parser.parse_args()
    
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

    # create model and load weights
    model = MultiFrameModelRecOnly(cfg=cfg)
    model.load_weights(cfg.model.weights)
    
    # create inference for model compilation
    input_img = tf.convert_to_tensor(np.zeros((cfg.training.batch_size, cfg.data.append, 4608), dtype=np.float32))
    _ = model(input_img)

    # save model with graph as pb
    tf.saved_model.save(model, "mfm_reconly")

    # use this command line to convert to onnx:
    # python3 -m tf2onnx.convert --saved-model mfm_reconly --output "model_reconly.onnx"
    