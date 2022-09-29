import os
import json
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess as sp
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from trainer.models.c3d import C3DModel
from trainer.models.classic import EchoFlexDownsampler, PerforDownsampler
from trainer.models.deepnet import DeepNetModel
from trainer.models.mf import MultiFrameModel
from trainer.models.vae import VAEModel
from trainer.models.mvit import MViTModel
from trainer.models.unet import UNeTModel
from trainer.models.utf import UNeTransformerModel
from data.preprocess import get_piano_notes_idxs

class ConfigManager(object):

    def __init__(self, json_name, random_seed=None, num_train=None, gpu=None, retrain=False):
        super(ConfigManager, self).__init__()

        # load json file
        self.json_path = os.path.join(os.getcwd(), json_name)
        with open(self.json_path) as f:
            cfg_dict = json.load(f)

        # add main properties
        self.retrain = retrain
        self.output_dir = cfg_dict['OUTPUT_DIR']
        self.store_csv = cfg_dict['STORE_CSV']
        if 'IS_TRAINING' in cfg_dict:
            self.is_training = cfg_dict['IS_TRAINING']
        else:
            self.is_training = True

        # random seed, to make all experiments equal
        if random_seed is not None: # override random seed
            self.random_seed = random_seed
        elif 'RANDOM_SEED' in cfg_dict:
            self.random_seed = cfg_dict['RANDOM_SEED']
        else:
            self.random_seed = None

        # set generalization study 
        if num_train is not None:
            self.num_train = num_train
            # override output dir
            self.output_dir = self.output_dir + "_ne" + str(self.num_train) + "_s" + str(self.random_seed)
        else:
            self.num_train = None

        # main task to perform, if unfilled, used regular
        if 'MODE' in cfg_dict:
            self.mode = cfg_dict['MODE']
        else:
            self.mode = 'regular'

        # define usage of other device in recordings (piano or keyboard)
        if self.mode == 'us2con2midi' or self.mode == 'typing' or self.mode == 'midi2midi' or self.mode == 'us2midi' or self.mode == 'us2multimidi' or self.mode == 'multimidi2multimidi' or self.mode == 'us2confNmultimidi' or \
           self.mode == 'flow2multimidi' or self.mode == 'flow2confNmultimidi' or self.mode == 'flow2multikey' or \
           self.mode == 'us2multikey' or self.mode == 'us2confNmultikey' or \
           self.mode == 'us2conf2multimidi' or self.mode == 'us2conf2multikey':
            self.use_dev = True
        else:
            self.use_dev = False

        # define usage of additional device input (positions and derivatives)
        if self.mode == 'us2con2midi' or self.mode == 'typing' or self.mode == 'midi2midi' or self.mode == 'us2midi':
            self.use_dev_x = True
        else:
            self.use_dev_x = False

        # define usage of flow maps
        if self.mode == 'flow2multimidi' or self.mode == 'flow2confNmultimidi' or self.mode == 'flow2multikey':
            self.use_flows = True
        else:
            self.use_flows = False

        # define usage of images
        if self.mode == 'midi2midi' or self.mode == 'multimidi2multimidi' or self.use_flows:
            self.use_imgs = False
        else:
            self.use_imgs = True

        # define usage of configuration space domain
        if self.mode == 'regular' or self.mode == 'us2conf' or self.mode == 'us2con2midi' or self.mode == 'us2usNconf' or self.mode == 'us2confNmultimidi' or self.mode == 'flow2confNmultimidi' or self.mode == 'us2confNmultikey' or \
           self.mode == 'us2conf2multimidi' or self.mode == 'us2conf2multikey':
            self.use_conf = True
        else:
            self.use_conf = False

        # main task to perform, if unfilled, used regular
        if 'OUTPUT_NAME' in cfg_dict:
            self.output_name = cfg_dict['OUTPUT_NAME']
        else:
            self.output_name = ""

        # add sub-config structures
        self.data = ConfigDataManager(cfg_dict=cfg_dict, retrain=retrain)
        self.model = ConfigModelManager(cfg_dict=cfg_dict)
        self.training = ConfigTrainingManager(cfg_dict=cfg_dict)
        self.system = ConfigSystemManager(cfg_dict=cfg_dict, gpu=gpu)


class ConfigDataManager(object):

    def __init__(self, cfg_dict, retrain):
        super(ConfigDataManager, self).__init__()

        # create sub-config structure
        self.path = cfg_dict['DATA']['PATH']
        self.train_files = cfg_dict['DATA']['TRAIN_FILES']
        self.test_files = cfg_dict['DATA']['TEST_FILES']

        # unsupervised files - add empty list if missing
        if 'UNSUPERVISED_FILES' in cfg_dict['DATA']:
            self.unsupervised_files = cfg_dict['DATA']['UNSUPERVISED_FILES']
        else:
            self.unsupervised_files = []

        self.joints_version = cfg_dict['DATA']['JOINTS_VERSION']

        # add joint names for custom
        if 'JOINT_NAMES' in cfg_dict['DATA']:
            self.joint_names = cfg_dict['DATA']['JOINT_NAMES']
        else:
            self.joint_names = []

        self.step = cfg_dict['DATA']['STEP']
        self.stride = cfg_dict['DATA']['STRIDE']

        # temporal - set if to ask for a bunch of images in one sample
        if 'TEMPORAL' in cfg_dict['DATA']:
            self.temporal = cfg_dict['DATA']['TEMPORAL']
        elif cfg_dict['MODEL']['TYPE'] == 'c3d' or cfg_dict['MODEL']['TYPE'] == 'c3de' or retrain:
            self.temporal = True
        else:
            self.temporal = False
        
        # sequence - set if to ask for a bunch of images in one sample
        if 'SEQUENCE' in cfg_dict['DATA']:
            self.sequence = cfg_dict['DATA']['SEQUENCE']
        elif cfg_dict['MODEL']['TYPE'] == 'basecorr' and retrain:
            self.sequence = True
        else:
            self.sequence = False

        # if to add the option of arm data
        if 'ARM' in cfg_dict['DATA']:
            self.arm = cfg_dict['DATA']['ARM']
        else:
            self.arm = False
        
        # set number of images to append
        if 'RES' in cfg_dict['DATA']:
            self.res = cfg_dict['DATA']['RES']
        elif cfg_dict['MODEL']['TYPE'] == 'c3d' or cfg_dict['MODEL']['TYPE'] == 'c3de':
            self.res = 112
        else:
            self.res = 224

        # set number of images to append
        if 'APPEND' in cfg_dict['DATA']:
            self.append = cfg_dict['DATA']['APPEND']
        elif cfg_dict['MODEL']['TYPE'] == 'c3d' or cfg_dict['MODEL']['TYPE'] == 'c3de':
            self.append = 16
        else:
            self.append = 1

        # set number of color channels
        if 'CHANNELS' in cfg_dict['DATA']:
            self.channels = cfg_dict['DATA']['CHANNELS']
        elif cfg_dict['MODEL']['TYPE'] == 'basecorrv2':
            self.channels = 3
        else:
            self.channels = 1

        # normalization method
        if 'NORMALIZATION' in cfg_dict['DATA']:
            self.normalization = cfg_dict['DATA']['NORMALIZATION']
        else:
            self.normalization = 'min_max'

        # derivative options
        if 'DERIVED' in cfg_dict['DATA']:
            self.derived = cfg_dict['DATA']['DERIVED']
        else:
            self.derived = False

        # robotic losses
        if 'FK' in cfg_dict['DATA']:
            self.fk = cfg_dict['DATA']['FK']
            self.use_fk = True
        else:
            self.fk = None
            self.use_fk = False
        if 'CD' in cfg_dict['DATA']:
            self.cd = cfg_dict['DATA']['CD']
            self.use_cd = True
        else:
            self.cd = None
            self.use_cd = False
        if 'FD' in cfg_dict['DATA']:
            self.fd = cfg_dict['DATA']['FD']
            self.use_fd = True
        else:
            self.fd = None
            self.use_fd = False

        # optical flow data
        if 'FLOW' in cfg_dict['DATA']:
            self.flow = cfg_dict['DATA']['FLOW']
        else:
            self.flow = False

        # if to give weights to samples before training
        if 'WEIGHTED_SAMPLING' in cfg_dict['DATA']:
            self.weighted_sampling = cfg_dict['DATA']['WEIGHTED_SAMPLING']
        else:
            self.weighted_sampling = False

        # if to share part of the data from test with train. choose 0.0 for no sharing
        if 'SHARE_TRAIN' in cfg_dict['DATA']:
            self.share_train = cfg_dict['DATA']['SHARE_TRAIN']
        else:
            self.share_train = 0.0

        # if to share a different fold of the data, rather than the last one
        if 'KFOLD' in cfg_dict['DATA']:
            self.kfold = cfg_dict['DATA']['KFOLD']
        else:
            self.kfold = None

        # if to share part of the data from test with train. choose 0.0 for no sharing
        if 'LEAVE_OUT_TEST' in cfg_dict['DATA']:
            self.leave_out_test = cfg_dict['DATA']['LEAVE_OUT_TEST']
        else:
            self.leave_out_test = 0.0

        # decide if single test it is or not
        if 'SINGLE_TEST' in cfg_dict['DATA']:
            self.single_test = cfg_dict['DATA']['SINGLE_TEST']
        else:
            self.single_test = False

        # decide if to shuffle data or not
        if 'SHUFFLE' in cfg_dict['DATA']:
            self.shuffle = cfg_dict['DATA']['SHUFFLE']
        elif self.single_test:
            self.shuffle = False
        else:
            self.shuffle = True

        # device related settings
        if 'DEV_CLASSES' in cfg_dict['DATA']:
            self.dev_classes = cfg_dict['DATA']['DEV_CLASSES']
        else:
            self.dev_classes = None
            
class ConfigModelManager(object):

    def __init__(self, cfg_dict):
        super(ConfigModelManager, self).__init__()

        # create sub-config structure
        self.type = cfg_dict['MODEL']['TYPE']
        
        # type of backbone, none for non-optional architecture
        if 'BACKBONE' in cfg_dict['MODEL']:
            self.backbone = cfg_dict['MODEL']['BACKBONE']
        else:
            self.backbone = None

        # version of backbone
        if 'BB_VERSION' in cfg_dict['MODEL']:
            self.bb_version = cfg_dict['MODEL']['BB_VERSION']
        else:
            self.bb_version = 'v1'

        # number of filters in backbone 
        if 'N_FILTERS' in cfg_dict['MODEL']:
            self.n_filters = cfg_dict['MODEL']['N_FILTERS']
        else:
            self.n_filters = 32
        
        # path to model weights - for retraining
        if 'WEIGHTS' in cfg_dict['MODEL']:
            self.weights = cfg_dict['MODEL']['WEIGHTS']
        else:
            self.weights = None

        # path to model weights - for retraining
        if 'Z_SIZE' in cfg_dict['MODEL']:
            self.z_size = cfg_dict['MODEL']['Z_SIZE']
        else:
            self.z_size = None

        # if to add residual layers or not
        if 'RES_LAYER' in cfg_dict['MODEL']:
            self.res_layer = cfg_dict['MODEL']['RES_LAYER']
        else:
            self.res_layer = False

                # if to add residual layers or not
        if 'CLASSIFIER' in cfg_dict['MODEL']:
            self.classifier = cfg_dict['MODEL']['CLASSIFIER']
        else:
            self.classifier = None


class ConfigTrainingManager(object):

    def __init__(self, cfg_dict):
        super(ConfigTrainingManager, self).__init__()

        # create sub-config structure
        # number of samples in one batch
        if 'BATCH_SIZE' in cfg_dict['TRAINING']:
            self.batch_size = cfg_dict['TRAINING']['BATCH_SIZE']
        else:
            self.batch_size = 32

        # number of epochs for training
        if 'EPOCHS' in cfg_dict['TRAINING']:
            self.epochs = cfg_dict['TRAINING']['EPOCHS']
        else:
            self.epochs = 40

        # number of epochs between each checkpoint saving
        if 'CP_INTERVAL' in cfg_dict['TRAINING']:
            self.cp_interval = cfg_dict['TRAINING']['CP_INTERVAL']
        else:
            self.cp_interval = 20

        # learning rate for optimizer
        if 'LEARNING_RATE' in cfg_dict['TRAINING']:
            self.learning_rate = cfg_dict['TRAINING']['LEARNING_RATE']
        else:
            self.learning_rate = 1e-3

class ConfigSystemManager(object):

    def __init__(self, cfg_dict, gpu=None):
        super(ConfigSystemManager, self).__init__()

        # create sub-config structure
        # gpu index to execute session on
        if gpu is not None:
            self.gpu = gpu
        elif 'GPU' in cfg_dict['SYSTEM']:
            self.gpu = cfg_dict['SYSTEM']['GPU']
        else:
            self.gpu = 0

        # memory requirement. if not used, 0 will not check for available memory in gpu
        if 'MEMORY_REQ' in cfg_dict['SYSTEM']:
            self.memory_req = cfg_dict['SYSTEM']['MEMORY_REQ']
        else:
            self.memory_req = 0

# set random seed in all platforms
def set_random_seed(cfg):
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
        tf.random.set_seed(seed=cfg.random_seed)
        np.random.seed(seed=cfg.random_seed)

# wait for gpu if asked
def wait_for_gpu(gpu, memory_req):

    if memory_req > 0:
        while get_gpu_memory(gpu) < memory_req:
            print('Waiting for gpu #{}...'.format(gpu))
            time.sleep(1)

# request free gpu memory from system
def get_gpu_memory(gpu):

    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    #print(memory_free_values)

    return memory_free_values[int(gpu)]

# create model based on config choice
def initiate_model(cfg):

    if cfg.model.type == 'c3d':
        return C3DModel(cfg=cfg, enhanced=False, n_filters=cfg.model.n_filters)
    elif cfg.model.type == 'c3de':
        return C3DModel(cfg=cfg, enhanced=True, n_filters=cfg.model.n_filters)
    elif cfg.model.type == 'vae':
        return VAEModel(cfg=cfg)
    elif cfg.model.type == 'mfm':
        return MultiFrameModel(cfg=cfg)
    elif cfg.model.type == 'mvit':
        return MViTModel(cfg=cfg)
    elif cfg.model.type == 'unet':
        return UNeTModel(cfg=cfg)
    elif cfg.model.type == 'utf':
        return UNeTransformerModel(cfg=cfg)
    elif cfg.model.type == 'deepnet':
        return DeepNetModel(cfg=cfg)
    elif cfg.model.type == 'echoflex':
        return EchoFlexDownsampler(cfg=cfg)
    elif cfg.model.type == 'perfor':
        return PerforDownsampler(cfg=cfg)
    else:
        return None

# create sklearn classifier based on config choice
def initiate_classifier(cfg):

    if cfg.model.classifier == 'svc':
        return SVC()
    elif cfg.model.classifier == 'mlp':
        return MLPClassifier()    
    elif cfg.model.classifier == 'rf':
        return RandomForestClassifier()
    elif cfg.model.classifier == 'lda':
        return LinearDiscriminantAnalysis()
    else:
        return None

# manage data loader output according to training mode
def filter_batch(batch, cfg):

    # get timestamp
    timestamp = batch[0]
    batch = batch[1:]

    # initialize variables
    img = y_flow = dev_x = y_conf = y_dev = None

    # get image or flow map
    if cfg.use_imgs or cfg.use_flows:
        img = batch[0]
        batch = batch[1:]

    # add arm positional data
    if cfg.data.arm: 
        dev_x = batch[0]
        batch = batch[1:]

    # add joints values
    if cfg.use_conf:
        y_conf = batch[0]
        batch = batch[1:]

    # add device key pressing probabilities
    if cfg.use_dev:
        y_dev = batch[0]
        batch = batch[1:]       

    return timestamp, img, dev_x, y_conf, y_dev 

# compute loss stats for configuration prediction
def compute_conf_metrics(conf_y, preds, timestamps, ike, fke, ide, fde, is_sequence):

    # compute the four metrics of the configuration estimation
    # inverse kinemaics
    ik_val = ike(conf_y, preds)
    ik_errors = tf.squeeze(tf.square(tf.subtract(tf.cast(conf_y, preds.dtype), preds) * np.pi), axis=0)

    # forward kinemaics
    if is_sequence:
        fk_val = fke(conf_y, preds)
        fk_errors = tf.squeeze(fke.fk_error, axis=0)

        # inverse dynamics
        ide.set_time(timestamps)
        id_val = ide(conf_y, preds)

        # forward dynamics
        fde.set_time_and_grads(timestamps, ide.y_true_grad, ide.y_pred_grad)
        fd_val = fde(conf_y, preds)

        return ik_val, fk_val, id_val, fd_val, ik_errors, fk_errors
    else:
        return ik_val, 0, 0, 0, ik_errors, 0

def compute_dev_metrics_per_class(dev_y, predictions, sequence):

    classes_num = dev_y.shape[-1]
    metrics_np = np.zeros((classes_num, 4))

    for i in range(classes_num):
        metrics_np[i,:] = compute_dev_metrics(dev_y=dev_y[:,:,i:i+1], predictions=predictions[:,:,i:i+1], sequence=sequence)

    return metrics_np

# compute confusion matrix stats
def compute_dev_metrics(dev_y, predictions, sequence=True):

    # compute idxs of predictions and labels, and transform to single element tensors
    if sequence:
        true_idxs = tf.map_fn(transform_idxs_to_unique_seq, tf.where(tf.math.greater(tf.cast(dev_y, tf.float32), tf.constant(0.5))))
        pred_idxs = tf.map_fn(transform_idxs_to_unique_seq, tf.where(tf.math.greater(predictions, tf.constant(0.5))))
    else:
        true_idxs = tf.map_fn(transform_idxs_to_unique, tf.where(tf.math.greater(tf.cast(dev_y, tf.float32), tf.constant(0.5))))
        pred_idxs = tf.map_fn(transform_idxs_to_unique, tf.where(tf.math.greater(predictions, tf.constant(0.5))))
    all_idxs = tf.concat([true_idxs,pred_idxs], axis=0)
    shared_idxs = tf.unique(all_idxs)[0]

    # compute confusion matrix count - tp, tn, fp, fn
    all_count = tf.size(dev_y)
    tp_count = tf.size(all_idxs) - tf.size(shared_idxs)
    tn_count = all_count - tf.size(shared_idxs)
    fp_count = tf.size(true_idxs) - tp_count
    fn_count = tf.size(pred_idxs) - tp_count

    # compute final metrics
    acc_val = tf.cast((tp_count + tn_count) / all_count, dtype=tf.float32)
    # if (tp_count + fn_count) > 0:
    rec_val = tf.cast(tp_count / (tp_count + fn_count), dtype=tf.float32)
    # else:
    #     rec_val = 1.0
    # if (tp_count + fp_count) > 0:
    pre_val = tf.cast(tp_count / (tp_count + fp_count), dtype=tf.float32)
    # else:
    #     pre_val = 1.0
    f1_val = tf.cast((2 * pre_val * rec_val) / (pre_val + rec_val), dtype=tf.float32)

    return acc_val, rec_val, pre_val, f1_val

def transform_idxs_to_unique(idx_tensor):

    return idx_tensor[1] + 100 * idx_tensor[0]

def transform_idxs_to_unique_seq(idx_tensor):

    return idx_tensor[2] + 100 * idx_tensor[1] + 10000 * idx_tensor[0]

def store_metrics_per_class(metrics_classes_np, data_dir):

    # prepare the three metrics in different dataframes
    classes = ['1','2','3','4','5']
    acc_df = pd.DataFrame(data=metrics_classes_np[:,:,0], columns=classes)
    rec_df = pd.DataFrame(data=metrics_classes_np[:,:,1], columns=classes)
    pre_df = pd.DataFrame(data=metrics_classes_np[:,:,2], columns=classes)
    f1_df = pd.DataFrame(data=metrics_classes_np[:,:,3], columns=classes)

    # store them
    acc_df.to_csv(os.path.join(data_dir, f'metrics_classes_acc.csv'), index=False)
    rec_df.to_csv(os.path.join(data_dir, f'metrics_classes_rec.csv'), index=False)
    pre_df.to_csv(os.path.join(data_dir, f'metrics_classes_pre.csv'), index=False)
    f1_df.to_csv(os.path.join(data_dir, f'metrics_classes_f1.csv'), index=False)

    # classes_num = metrics_classes_np.shape[1]
    # metrics_np_states = np.zeros((classes_num, 4))
    # for i in range(classes_num):
    #     acc_mean = metrics_classes_np[:,i,0][~np.isnan(metrics_classes_np[:,i,0])].mean()
    #     rec_mean = metrics_classes_np[:,i,1][~np.isnan(metrics_classes_np[:,i,1])].mean()
    #     pre_mean = metrics_classes_np[:,i,2][~np.isnan(metrics_classes_np[:,i,2])].mean()
    #     f1_mean = metrics_classes_np[:,i,3][~np.isnan(metrics_classes_np[:,i,3])].mean()

    #     metrics_np_states[i,:] = [acc_mean, rec_mean, pre_mean, f1_mean]

    # # save metrics dataframe
    # metrics_df = pd.DataFrame(data=metrics_np_states, columns=['acc','rec','pre','f1'])
    # df_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    # metrics_df.to_csv(f'metrics_{df_time}.csv', index=False)

def store_raw_predictions_and_images(cfg, images, labels, predictions, output_dir=None, store_imgs=True, classes=None, arm_lengths=None, palm_vectors=None):

    # prepare output folder
    dir_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if output_dir is None:
        dir_name = os.path.join(cfg.output_dir, f'raw_exp_data_{dir_time}')
    else:
        dir_name = os.path.join(output_dir, f'raw_exp_data')
    os.makedirs(dir_name)

    # denormalize images and store them
    if store_imgs:
        imgs_dir_name = os.path.join(dir_name, 'images')
        os.makedirs(imgs_dir_name)
        images = np.uint8(images * 255)
        for i in range(len(images)):
            cv2.imwrite(os.path.join(imgs_dir_name, f'{i}.png'), images[i])

    # store labels and predictions
    if cfg.mode == 'us2conf': # plot configurations
        preds_df = pd.DataFrame(data=predictions, columns=classes)
        labels_df = pd.DataFrame(data=labels, columns=classes)
    else:
        classes = ['01', '02', '03', '04', '05']
        preds_df = pd.DataFrame(data=np.where(predictions > 0.5, 1, 0), columns=classes)
        labels_df = pd.DataFrame(data=np.where(labels > 0.5, 1, 0), columns=classes)
    preds_df.to_csv(os.path.join(dir_name, 'preds.csv'), index=True)
    labels_df.to_csv(os.path.join(dir_name, 'labels.csv'), index=True)

    # save arm lengths for conf visualization
    if arm_lengths is not None:
        arm_df = pd.DataFrame(data=arm_lengths, columns=['link1', 'link2', 'link3'])
        arm_df.to_csv(os.path.join(dir_name, 'arm.csv'), index=True)
    if palm_vectors is not None:
        palm_df = pd.DataFrame(data=palm_vectors.T, columns=['thumb3', 'thumb5', 'finger11', 'finger21', 'finger31', 'finger41', 'finger12', 'finger22', 'finger32', 'finger42'])
        palm_df.to_csv(os.path.join(dir_name, 'palm.csv'), index=True)

def create_playing_visualization(images, preds, dev_y):

    print('Generating playing visualization...')

    # generate playing image
    notes_names = ['thumb', 'finger1', 'finger2', 'finger3', 'finger4']
    img, notes_polygons = create_notes_img(notes_names)

    # define video stream
    video_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    video = cv2.VideoWriter(os.path.join(os.getcwd(), f'video_{video_time}.avi'), 0, 19, (img.shape[0]*3,img.shape[1]))

    # iterate over predictions and create frames for vide
    for i in range(len(preds)):

        # create frame for image
        us_frame = cv2.cvtColor((images[i] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        us_frame = cv2.resize(us_frame, (180, 180)) 
        
        # create frame for predicted notes
        pred_frame = img.copy()
        pressed_idxs = np.where(preds[i] > 0.5)[0]
        for pressed_idx in pressed_idxs:
            pressed_idx
            notes_names[pressed_idx]
            pred_frame = cv2.fillPoly(pred_frame, pts=[notes_polygons[notes_names[pressed_idx]].reshape((-1, 1, 2))], color=(0,0,255))

        # create frame for ground-truth notes
        gt_frame = img.copy()
        pressed_idxs = np.where(dev_y[i] > 0.5)[0]
        for pressed_idx in pressed_idxs:
            pressed_idx
            notes_names[pressed_idx]
            gt_frame = cv2.fillPoly(gt_frame, pts=[notes_polygons[notes_names[pressed_idx]].reshape((-1, 1, 2))], color=(255,0,0))

        # concatenate frames and add to video
        vid_frame = np.concatenate([us_frame, pred_frame, gt_frame], axis=1)
        video.write(vid_frame)

    # store video
    video.release()
    cv2.destroyAllWindows()

def create_notes_img(notes_names):

    # dimensions in pixels
    note_height, note_length = 110, 22
    img_height, img_width = 180, 180
    
    # create image
    img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # define notes polygons for drawing
    notes_polygons = {}
    origin = np.array([35, 20], dtype=np.int32)
    for i, note_name in enumerate(notes_names):
        
        # create note polygon
        note_origin = origin + i*np.array([note_length, 0])
        note_polygon = np.array([note_origin, note_origin + np.array([0, note_height]), note_origin + np.array([note_length, note_height]), note_origin + np.array([note_length, 0])])
        
        # store it with caption
        notes_polygons[note_name] = note_polygon
        
    # draw notes on the image
    for note_name, note_polygon in notes_polygons.items():
        img = cv2.polylines(img, [note_polygon.reshape((-1, 1, 2))], True, (0,0,0), 1)

    return img, notes_polygons
    
def note_idx_to_note(note_idx):

    # define piano notes indices
    white_notes_idxs, black_notes_idxs = get_piano_notes_idxs()

    # define names and octaves
    wnotes_names = ['A','B','C','D','E','F','G'] * 7 + ['A','B','C']
    wnotes_octaves = 2*[0] + 7*[1] + 7*[2] + 7*[3] + 7*[4] + 7*[5] + 7*[6] + 7*[7] + [8]

    if note_idx in white_notes_idxs:
        return wnotes_names[white_notes_idxs.index(note_idx)], wnotes_octaves[white_notes_idxs.index(note_idx)]


def store_conf_metrics(data_dir, ik_errors, fk_errors, ik_labels, fk_labels):

    # create dataframes for ik and fk
    ik_errors_df = pd.DataFrame(data=ik_errors, columns=ik_labels)
    fk_errors_df = pd.DataFrame(data=fk_errors, columns=fk_labels)

    # save them
    ik_errors_df.to_csv(os.path.join(data_dir, 'metrics_conf_ik_df.csv'), index=False)
    fk_errors_df.to_csv(os.path.join(data_dir, 'metrics_conf_fk_df.csv'), index=False)


def create_confs_graph(preds_np, labels_np, labels_names):

    print('Plotting graphs...')

    n_rows, n_cols = 5, 4
    x = range(preds_np.shape[0])
    fig = plt.figure()
    fig.set_size_inches(20,16)
    fig.patch.set_facecolor('white')

    for i, label_name in enumerate(labels_names):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(x, labels_np[:,i], color='b')
        plt.plot(x, preds_np[:,i], color='r')
        plt.title(label_name)
        plt.subplots_adjust(hspace=1.0)
        plt.grid()

    graph_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    plt.savefig(os.path.join(os.getcwd(), 'graph_cs_{}.png'.format(graph_time)), dpi=300)

def store_predicted_images(images_np, predictions_np, output_dir):

    # prepare output folder
    dir_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    dir_name = os.path.join(output_dir, f'imgs_{dir_time}')
    os.makedirs(dir_name)

    # iterate over predictions and store image along each prediction
    for i in range(len(images_np)):

        # extract images as UINT8 arrays
        img_np = np.uint8(images_np[i,:,:] * 255)
        predimg_np = np.uint8(predictions_np[i,:,:] * 255)

        # concat and save both images as one file
        cv2.imwrite(os.path.join(dir_name, f'img_{i}.png'), np.concatenate([img_np, predimg_np], axis=1))
