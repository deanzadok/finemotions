import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import cv2
import pytesseract
import datetime
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tensorflow as tf

###################################################
#            Data Management Class                #
#                                                 #
# 1. Load h5 files must be stored in train and    #
# test folders under the data_dir.                #
# 2. Mode must be delivered in order to get       #
# ground truth.                                   #
###################################################

class DataManagement(object):

    def __init__(self, cfg, brighten_range=0.4):
        super(DataManagement, self).__init__()

        # init parameters
        self.cfg = cfg
        self.brighten_range = brighten_range

        # load and preprocess data
        self.load_data()
        self.dilute_data()
        self.crop_and_resize_images()

        # change images format to numpy
        self.transform_images_to_arrays()

        self.define_ground_truth()
        if self.cfg.data.use_fk or self.cfg.data.use_fd: # prepare arm lengths for forward kinematics loss function
            self.compute_arm_lengths()
        self.prepare_samples()
        if self.cfg.use_dev:
            self.prepare_device_data()
        else:
            self.remove_device_data()
        if self.cfg.data.weighted_sampling:
            self.prepare_samples_weighting()
        self.shuffle_data()

        if not self.cfg.data.single_test:
            print('Total samples for train: {}.\nTotal samples for test: {}.'.format(len(self.train_df),len(self.test_df)))
        else:
            print('Total samples for single test: {}.'.format(len(self.test_df)))

        # perform normalization
        self.normalize_data()
        
        # just a sanity check
        self.remove_nan_values()
        
        # generator output types
        if self.cfg.mode == 'us2conf2multimidi' or self.cfg.mode == 'us2conf2multikey':
            output_types = (tf.float64, tf.float64, tf.float64, tf.int32)
        elif self.cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2multikey':
            output_types = (tf.float64, tf.int32)
        else:
            output_types = (tf.float64, tf.float64, tf.float64)

        # create generators
        self.train_gen = tf.data.Dataset.from_generator(self.train_generator, output_types=output_types).batch(self.cfg.training.batch_size)
        self.test_gen = tf.data.Dataset.from_generator(self.test_generator, output_types=output_types).batch(self.cfg.training.batch_size)

    # load all the h5 files for training and testing
    def load_data(self):

        print('Loading dataset...')

        train_dfs, test_dfs = [], []
        # load train h5 files
        for h5_file in [os.path.join(self.cfg.data.path, x) for x in self.cfg.data.train_files]:
            # check if file exist
            if os.path.isfile(h5_file):
                train_dfs.append(pd.read_hdf(path_or_buf=h5_file,key='df'))
                train_dfs[-1]['sample_index'] = train_dfs[-1].index
            else:
                print(f'{h5_file} was not found!')
        # load test hf files. make sure to load only one of them if asked
        for h5_file in [os.path.join(self.cfg.data.path, x) for x in self.cfg.data.test_files]:
            if os.path.isfile(h5_file):
                test_dfs.append(pd.read_hdf(path_or_buf=h5_file,key='df'))
                test_dfs[-1]['sample_index'] = test_dfs[-1].index
            else:
                print(f'{h5_file} was not found!')

        # share test sessions with train sessions by moving part of each dataframe to train if asked
        if self.cfg.data.share_train > 0.0:

            # define k-fold
            if self.cfg.data.kfold is not None:
                k_fold_size = 1 - self.cfg.data.share_train

                for i in range(len(test_dfs)):
                    # define fold start and end indices
                    fold_start_idx = int(self.cfg.data.kfold * (k_fold_size * len(test_dfs[i])))
                    fold_end_idx = int((self.cfg.data.kfold + 1) * (k_fold_size * len(test_dfs[i])))

                    # separate training samples and add them to the list
                    train_df_bfold = test_dfs[i].iloc[:fold_start_idx]
                    train_df_afold = test_dfs[i].iloc[fold_end_idx:]
                    if len(train_df_bfold) > 0:
                        train_dfs.append(train_df_bfold)
                    if len(train_df_afold) > 0:
                        train_dfs.append(train_df_afold)

                    test_dfs[i] = test_dfs[i].iloc[fold_start_idx:fold_end_idx]
            else:
                for i in range(len(test_dfs)):
                    train_dfs.append(test_dfs[i].iloc[:int(len(test_dfs[i]) * self.cfg.data.share_train)])
                    test_dfs[i] = test_dfs[i].iloc[int(len(test_dfs[i]) * self.cfg.data.share_train):]

        # leave samples out of test session if asked
        if self.cfg.data.leave_out_test > 0.0:
            for i in range(len(test_dfs)):
                test_dfs[i] = test_dfs[i].iloc[int(len(test_dfs[i]) * self.cfg.data.leave_out_test):]

        # concatenate them into single dataframes
        if not self.cfg.data.single_test:
            self.train_df = pd.concat(train_dfs)
        else:
            self.train_df = pd.DataFrame()
        self.test_df = pd.concat(test_dfs)

    # dilute data by sekecting only samples after strides
    def dilute_data(self):
        if self.cfg.data.stride > 1:
            if not self.cfg.data.single_test:
                self.train_df = self.train_df.iloc[::self.cfg.data.stride,:]
            self.test_df = self.test_df.iloc[::self.cfg.data.stride,:]

    # resize images in dataset, after cropping them if they were not squared
    def crop_and_resize_images(self):

        if self.cfg.use_imgs:
            print('Cropping and resizing images...')

            # crop image if needed - width
            if len(self.train_df) > 0:
                img_size = self.train_df.img.iloc[0].size
            else:
                img_size = self.test_df.img.iloc[0].size
            if img_size[0] > img_size[1]:
                offset = img_size[0] - img_size[1]
                if not self.cfg.data.single_test:
                    self.train_df['img'] = self.train_df['img'].apply(lambda x: x.crop((offset//2, 0, img_size[0] - offset//2, img_size[1])))
                self.test_df['img'] = self.test_df['img'].apply(lambda x: x.crop((offset//2, 0, img_size[0] - offset//2, img_size[1])))
            
            # crop image if needed - height
            if img_size[0] < img_size[1]:
                offset = img_size[1] - img_size[0]
                if not self.cfg.data.single_test:
                    self.train_df['img'] = self.train_df['img'].apply(lambda x: x.crop((0, offset//2, img_size[0], img_size[1] - offset//2)))
                self.test_df['img'] = self.test_df['img'].apply(lambda x: x.crop((0, offset//2, img_size[0], img_size[1] - offset//2)))

            # resize image if asked
            if self.cfg.data.res > 0:
                if not self.cfg.data.single_test:
                    self.train_df['img'] = self.train_df['img'].apply(lambda x: x.resize((self.cfg.data.res,self.cfg.data.res)))
                self.test_df['img'] = self.test_df['img'].apply(lambda x: x.resize((self.cfg.data.res,self.cfg.data.res)))
        else:
            print('Removing images...')

            # remove images from all dataframes in case training does not require images
            if not self.cfg.data.single_test:
                self.train_df.drop(columns=['img'], inplace=True)
            self.test_df.drop(columns=['img'], inplace=True)

    def compute_arm_lengths(self):
        # compute mean arm lengths
        if self.cfg.data.joints_version == '3' or self.cfg.data.joints_version == '4':
            self.arm_lengths = np.array([[self.get_arm_length('finger41','finger42'), self.get_arm_length('finger42','finger43'), self.get_arm_length('finger43','finger44')],
                                        [self.get_arm_length('finger31','finger32'), self.get_arm_length('finger32','finger33'), self.get_arm_length('finger33','finger34')],
                                        [self.get_arm_length('finger21','finger22'), self.get_arm_length('finger22','finger23'), self.get_arm_length('finger23','finger24')],
                                        [self.get_arm_length('finger11','finger12'), self.get_arm_length('finger12','finger13'), self.get_arm_length('finger13','finger14')],
                                        [self.get_arm_length('thumb5','thumb6'), self.get_arm_length('thumb6','thumb7'), 0.0]])
        else: # self.cfg.data.joints_version == 1:
            self.arm_lengths = np.array([[self.get_arm_length('finger41','finger42'), self.get_arm_length('finger42','finger43')],
                                        [self.get_arm_length('finger31','finger32'), self.get_arm_length('finger32','finger33')],
                                        [self.get_arm_length('finger21','finger22'), self.get_arm_length('finger22','finger23')],
                                        [self.get_arm_length('finger11','finger12'), self.get_arm_length('finger12','finger13')],
                                        [self.get_arm_length('thumb2','thumb3'), self.get_arm_length('thumb3','thumb4')]])

    # compute mean arm length given two link names
    def get_arm_length(self, link_1, link_2):
        if not self.cfg.data.single_test:
            return self.train_df.apply(lambda x: np.linalg.norm(x[link_1] - x[link_2]), axis=1).mean()
        return self.test_df.apply(lambda x: np.linalg.norm(x[link_1] - x[link_2]), axis=1).mean()

    # append images and labels to create stacked inputs and outputs
    def prepare_samples(self):

        print('Preparing samples...')

        # prepare columns for concatenation
        concatenated_labels = self.labels_names.copy()
        if self.cfg.use_imgs:
            concatenated_labels += ['img']
        if not self.cfg.data.single_test:
            self.train_df[concatenated_labels] = self.train_df[concatenated_labels].applymap(lambda x: [x])
        self.test_df[concatenated_labels] = self.test_df[concatenated_labels].applymap(lambda x: [x])

        # create new column for stacked images
        if self.cfg.use_imgs:
            if not self.cfg.data.single_test:
                self.train_df['imgs'] = self.train_df['img']
            self.test_df['imgs'] = self.test_df['img']

        # define labels names for stacked and derived output
        self.labels_names_stacked = [x+'s' for x in self.labels_names]

        # create timestamp as x-space for gradient computation
        if not self.cfg.data.single_test:
            self.train_df['timestamp'] = (self.train_df.datetime.astype(int) / 1e6).astype(int).apply(lambda x: [x])
        self.test_df['timestamp'] = (self.test_df.datetime.astype(int) / 1e6).astype(int).apply(lambda x: [x])

        # create new columns for stacked ground truth
        if len(self.labels_names_stacked) > 0:
            if not self.cfg.data.single_test:
                self.train_df[self.labels_names_stacked] = self.train_df[self.labels_names]
                train_df_temp = self.train_df.copy()
            self.test_df[self.labels_names_stacked] = self.test_df[self.labels_names]
            test_df_temp = self.test_df.copy()

        # shift by the step size to append samples together
        for i in range(self.cfg.data.step,self.cfg.data.append*self.cfg.data.step,self.cfg.data.step):
            # append images
            if self.cfg.use_imgs:
                if not self.cfg.data.single_test:
                    self.train_df['imgs'] = self.train_df.shift(i)['img'] + self.train_df['imgs']
                self.test_df['imgs'] = self.test_df.shift(i)['img'] + self.test_df['imgs']

            # append labels
            if len(self.labels_names_stacked) > 0:
                if not self.cfg.data.single_test:
                    self.train_df[self.labels_names_stacked] = train_df_temp.shift(i)[self.labels_names_stacked] + self.train_df[self.labels_names_stacked]
                self.test_df[self.labels_names_stacked] = test_df_temp.shift(i)[self.labels_names_stacked] + self.test_df[self.labels_names_stacked]

            # append timestamps
            if not self.cfg.data.single_test:
                self.train_df['timestamp'] = train_df_temp.shift(i)['timestamp'] + self.train_df['timestamp']
            self.test_df['timestamp'] = test_df_temp.shift(i)['timestamp'] + self.test_df['timestamp']

        # drop rows with missing information
        if not self.cfg.data.single_test:
            self.train_df = self.train_df.iloc[self.cfg.data.append*self.cfg.data.step-1:]
        self.test_df = self.test_df.iloc[self.cfg.data.append*self.cfg.data.step-1:]

        # convert labels to numpy for future computations
        np_labels = self.labels_names_stacked + ['timestamp']
        if not self.cfg.data.single_test:
            self.train_df[np_labels] = self.train_df[np_labels].applymap(lambda x: np.array(x))
        self.test_df[np_labels] = self.test_df[np_labels].applymap(lambda x: np.array(x))

    def prepare_device_data(self):

        # set device input names and choose label name based on requested device
        self.device_input_pos_names = ['thumb3_dev', 'thumb4_dev']
        self.device_input_vel_names = ['thumb3_der', 'thumb4_der']
        self.device_input_acc_names = ['thumb3_der_der', 'thumb4_der_der']
        self.device_input_names = self.device_input_pos_names + self.device_input_vel_names + self.device_input_acc_names

        if self.cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2conf2multimidi':
            self.device_label_name = 'notes_multi'
        else: # self.cfg.mode == 'us2multikey' or self.cfg.mode == 'us2conf2multikey':
            self.device_label_name = 'keys_multi'
        self.device_label_name_stacked = self.device_label_name + 's'
        self.device_input_pos_names_stacked = [x+'s' for x in self.device_input_pos_names]
        self.device_input_vel_names_stacked = [x+'s' for x in self.device_input_vel_names]
        self.device_input_acc_names_stacked = [x+'s' for x in self.device_input_acc_names]
        self.device_input_names_stacked = [x+'s' for x in self.device_input_names]

        # prepare columns for concatenation
        if not self.cfg.data.single_test:
            self.train_df[self.device_label_name] = self.train_df[self.device_label_name].apply(lambda x: [x])
            self.train_df[self.device_input_names] = self.train_df[self.device_input_names].applymap(lambda x: [x])
        self.test_df[self.device_label_name] = self.test_df[self.device_label_name].apply(lambda x: [x])
        self.test_df[self.device_input_names] = self.test_df[self.device_input_names].applymap(lambda x: [x])

        # create new column for stacked notes and backup dataframes
        if not self.cfg.data.single_test:
            self.train_df[self.device_label_name_stacked] = self.train_df[self.device_label_name]
            self.train_df[self.device_input_names_stacked] = self.train_df[self.device_input_names]
            train_df_temp = self.train_df.copy()
        self.test_df[self.device_label_name_stacked] = self.test_df[self.device_label_name]
        self.test_df[self.device_input_names_stacked] = self.test_df[self.device_input_names]
        test_df_temp = self.test_df.copy()

        # shift by the step size to append device labels together
        for i in range(self.cfg.data.step,self.cfg.data.append*self.cfg.data.step,self.cfg.data.step):
            # append device labels
            if not self.cfg.data.single_test:
                self.train_df[self.device_label_name_stacked] = train_df_temp.shift(i)[self.device_label_name_stacked] + self.train_df[self.device_label_name_stacked]
            self.test_df[self.device_label_name_stacked] = test_df_temp.shift(i)[self.device_label_name_stacked] + self.test_df[self.device_label_name_stacked]

            # append device inputs
            if not self.cfg.data.single_test:
                self.train_df[self.device_input_names_stacked] = train_df_temp.shift(i)[self.device_input_names_stacked] + self.train_df[self.device_input_names_stacked]
            self.test_df[self.device_input_names_stacked] = test_df_temp.shift(i)[self.device_input_names_stacked] + self.test_df[self.device_input_names_stacked]

        # drop the old column
        if not self.cfg.data.single_test:
            self.train_df = self.train_df.drop(columns=[self.device_label_name] + self.device_input_names)
        self.test_df = self.test_df.drop(columns=[self.device_label_name] + self.device_input_names)

        # backfill device label to save early samples
        self.device_data_names = [self.device_label_name_stacked] + self.device_input_names_stacked
        if not self.cfg.data.single_test:
            self.train_df[self.device_data_names] = self.train_df[self.device_data_names].fillna(method='bfill')
        self.test_df[self.device_data_names] = self.test_df[self.device_data_names].fillna(method='bfill')

        # convert device label to numpy array
        if not self.cfg.data.single_test:
            self.train_df[self.device_data_names] = self.train_df[self.device_data_names].applymap(lambda x: np.array(x))
        self.test_df[self.device_data_names] = self.test_df[self.device_data_names].applymap(lambda x: np.array(x))

    # don't keep what you don't need
    def remove_device_data(self):

        labels_to_remove = []
        if 'keys' in self.train_df.columns:
            labels_to_remove.append('keys')
        if 'keys_multi' in self.train_df.columns:
            labels_to_remove.append('keys_multi')
        if 'notes' in self.train_df.columns:
            labels_to_remove.append('notes')
        if 'notes_multi' in self.train_df.columns:
            labels_to_remove.append('notes_multi')

        # remove device labels
        if not self.cfg.data.single_test:
            self.train_df.drop(columns=labels_to_remove, inplace=True)
        self.test_df.drop(columns=labels_to_remove, inplace=True)

    # initiate column with weights for sampling that will be used in the train generator
    def prepare_samples_weighting(self):

        self.train_df['weight'] = self.train_df.apply(lambda x: 1 / (x[self.labels_names_stacked].mean()[-1]), axis=1)

    # shuffle datasets
    def shuffle_data(self):
        if not self.cfg.data.single_test and self.cfg.data.shuffle: # do not shuffle if single test is required
            self.train_df = self.train_df.sample(frac=1.0)
            self.test_df = self.test_df.sample(frac=1.0)

    # transform images from PIL images to numpy arrays
    def transform_images_to_arrays(self):

        if self.cfg.use_imgs:
            print('Transforming images back to arrays...')

            if not self.cfg.data.single_test:
                self.train_df.img = self.train_df.img.apply(lambda x: np.asarray(x))
            self.test_df.img = self.test_df.img.apply(lambda x: np.asarray(x))

    # extract labels as an array of shape [NC] for train and test
    def define_ground_truth(self):
        if self.cfg.data.joints_version == '0' or not self.cfg.use_conf: # other mode that not using joints
            self.labels_names = []
        if self.cfg.data.joints_version == '1s':
            self.labels_names = ['joint41', 'joint31', 'joint21', 'joint11']
        elif self.cfg.data.joints_version == '2':
            self.labels_names = ['joint42', 'joint32', 'joint22', 'joint12',\
                                 'joint43', 'joint33', 'joint23', 'joint13',\
                                 'joint44', 'joint34','joint24', 'joint14',\
                                 'jointt3', 'jointt4', 'jointwy']
        elif self.cfg.data.joints_version == '2c':
            self.labels_names = ['joint4234', 'joint3234', 'joint2234', 'joint1234',\
                                 'jointt34', 'jointwy']
        elif self.cfg.data.joints_version == '3':
            self.labels_names = ['joint41', 'joint31', 'joint21', 'joint11',\
                                 'joint42', 'joint32', 'joint22', 'joint12',\
                                 'joint43', 'joint33', 'joint23', 'joint13',\
                                 'jointt5', 'jointt6', 'wristy']
        elif self.cfg.data.joints_version == '4':
            self.labels_names = ['joint41', 'joint31', 'joint21', 'joint11',\
                                 'joint42', 'joint32', 'joint22', 'joint12',\
                                 'joint43', 'joint33', 'joint23', 'joint13',\
                                 'jointt5', 'jointt6',\
                                 'jointwr', 'jointwp', 'jointwy']
        elif self.cfg.data.joints_version == '1':
            self.labels_names = ['joint41', 'joint31', 'joint21', 'joint11', 'joint42','joint32',\
                                 'joint22', 'joint12', 'jointt2', 'jointt3', 'jointt1']
        else: # self.cfg.data.joints_version == 'custom'
            self.labels_names = self.cfg.data.joint_names

    # data normalization function. choose from ['min_max', 'z_score']
    # images normalization is done temporarily using the generators
    def normalize_data(self, auto=False, old_min_x=0.0, old_max_x=255.0, old_min_y=0.0, old_max_y=np.pi,\
                                                 new_min_x=0.0, new_max_x=1.0, new_min_y=0.0, new_max_y=1.0):
        
        print('Normalizing data...')

        if len(self.labels_names_stacked) > 0 and self.cfg.data.normalization == 'min_max':
        
            # define min-max ranges for Y
            if auto:
                if not self.cfg.data.single_test:
                    old_min_y = np.vstack(self.train_df[self.labels_names_stacked].stack().values).min()
                    old_max_y = np.vstack(self.train_df[self.labels_names_stacked].stack().values).max()
                else:
                    old_min_y = np.vstack(self.test_df[self.labels_names_stacked].stack().values).min()
                    old_max_y = np.vstack(self.test_df[self.labels_names_stacked].stack().values).max()     
            y_old_range = old_max_y - old_min_y
            y_new_range = new_max_y - new_min_y

            # normalize Y using min-max
            if not self.cfg.data.single_test:
                self.train_df[self.labels_names_stacked] = new_min_y + y_new_range * (self.train_df[self.labels_names_stacked] - old_min_y) / y_old_range
            self.test_df[self.labels_names_stacked] = new_min_y + y_new_range * (self.test_df[self.labels_names_stacked] - old_min_y) / y_old_range

        elif len(self.labels_names_stacked) > 0: # self.cfg.data.normalization == 'z_score':

            if not self.cfg.data.single_test:
                self.Y_train_mean = self.train_df[self.labels_names_stacked].values.mean().mean()
                self.Y_train_std = self.train_df[self.labels_names_stacked].values.std().mean()
                self.train_df[self.labels_names_stacked] = (self.train_df[self.labels_names_stacked] - self.Y_train_mean) / self.Y_train_std

                # save mean and std for later use
                stats_df = pd.DataFrame(data={'y_train_mean':[self.Y_train_mean],'y_train_std':[self.Y_train_std]})
                stats_df.to_csv(os.path.join(self.cfg.output_dir,"stats.csv"), index=False)
            else:
                # load stats if exists
                if os.path.isfile(os.path.join(self.cfg.output_dir,"stats.csv")):
                    stats_df = pd.read_csv(os.path.join(self.cfg.output_dir,"stats.csv"))
                    self.Y_train_mean = stats_df.y_train_mean.iloc[0]
                    self.Y_train_std = stats_df.y_train_std.iloc[0]
                else:
                    self.Y_train_mean = self.test_df[self.labels_names_stacked].values.mean().mean()
                    self.Y_train_std = self.test_df[self.labels_names_stacked].values.std().mean()
            self.test_df[self.labels_names_stacked] = (self.test_df[self.labels_names_stacked] - self.Y_train_mean) / self.Y_train_std

        # normalize device-related input
        if self.cfg.use_dev:

            piano_pos_coeffs = np.array([300, 980, 100])
            piano_acc_coeffs = (piano_pos_coeffs**2)/2

            if not self.cfg.data.single_test:

                # position information
                self.train_df[self.device_input_pos_names_stacked] = self.train_df[self.device_input_pos_names_stacked].applymap(lambda x: x/piano_pos_coeffs)

                # velocity information
                self.train_df[self.device_input_vel_names_stacked] = self.train_df[self.device_input_vel_names_stacked].applymap(lambda x: x/piano_pos_coeffs)

                # acceleration information
                self.train_df[self.device_input_acc_names_stacked] = self.train_df[self.device_input_acc_names_stacked].applymap(lambda x: x/piano_acc_coeffs)

            self.test_df[self.device_input_pos_names_stacked] = self.test_df[self.device_input_pos_names_stacked].applymap(lambda x: x/piano_pos_coeffs)
            self.test_df[self.device_input_vel_names_stacked] = self.test_df[self.device_input_vel_names_stacked].applymap(lambda x: x/piano_pos_coeffs)
            self.test_df[self.device_input_acc_names_stacked] = self.test_df[self.device_input_acc_names_stacked].applymap(lambda x: x/piano_acc_coeffs)

    # remove nan values. do nothing if there are no nan values
    def remove_nan_values(self):
        if not self.cfg.data.single_test:
            if len(self.train_df) > len(self.train_df.dropna()):
                print('WARNING: removed {} lines of train set'.format(len(self.train_df) - len(self.train_df.dropna())))
                self.train_df = self.train_df.dropna()
        self.test_df = self.test_df.dropna()

###################################################
#               Data Generators                   #
###################################################

    def train_generator(self):
        # yield image and label by iterating the data
        for sample_idx in range(len(self.train_df)):
            
            # draw sample
            if self.cfg.data.weighted_sampling:
                sample_values = self.train_df.sample(n=1, weights='weight').iloc[0]
            else:
                sample_values = self.train_df.iloc[sample_idx]

            # skip if sample does not hold images from the same recording session
            if sample_values.sample_index < self.cfg.data.append * self.cfg.data.step - 1:
                continue

            if self.cfg.use_imgs: # need for images
                # draw random state for augmentations
                max_cropped = int(self.cfg.data.res / 8)
                max_shift = int(self.cfg.data.res / 10)
                rand_state = [np.random.randint(low=0, high=2, size=2), # horizontal and vertical flips
                            np.random.uniform(low=1.0-self.brighten_range, high=1.0+self.brighten_range), # random brightening
                            np.random.randint(low=0.0-max_shift, high=0.0+max_shift), # random shifting
                            np.random.randint(low=0, high=4), # random rotations with multiples of 90s
                            np.random.randint(low=0, high=int(max_cropped/2), size=3), # random resize and crop
                            np.random.randint(low=0, high=125)]

                # append images
                imgs = []
                for j in range(len(sample_values['imgs'])):
                    # augment image using the random state and normalize
                    img = sample_values['imgs'][j]
                    if self.cfg.use_imgs:
                        img = self.random_augment(img, rand_state)
                    # convert to RGB if asked
                    if self.cfg.data.channels == 3:
                        imgs.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                    imgs.append(np.expand_dims(img,axis=-1))
                img = np.concatenate(imgs, axis=-1)

                # fill void space with noise
                if self.cfg.use_imgs:
                    if rand_state[2] < -1:
                        img[:,:-rand_state[2],:] = np.clip(np.rint(np.random.normal(loc=img[:,-rand_state[2]:,:].mean(), scale=img[:,-rand_state[2]:,:].std(), size=img[:,:-rand_state[2],:].shape)), a_min=0.0, a_max=255.0)
                    elif rand_state[2] > 1:
                        img[:,-rand_state[2]:,:] = np.clip(np.rint(np.random.normal(loc=img[:,:-rand_state[2],:].mean(), scale=img[:,:-rand_state[2],:].std(), size=img[:,-rand_state[2]:,:].shape)), a_min=0.0, a_max=255.0)

                # add gaussian noise
                img = img + (np.random.normal(loc=0.0, scale=np.sqrt(0.025), size=img.shape) * rand_state[5])
                img = np.clip(img, 0, 255)

            # yield inputs and labels for each mode
            if self.cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2multikey':
                img = img / 255.0
                dev_y = sample_values[self.device_label_name_stacked]
                if not self.cfg.data.sequence:
                    dev_y = dev_y[-1]
            
                yield img, dev_y            
            elif self.cfg.mode == 'us2conf2multimidi' or self.cfg.mode == 'us2conf2multikey':
                img = img / 255.0
                dev_y = sample_values[self.device_label_name_stacked]
                if self.cfg.data.sequence:
                    y = np.swapaxes(np.vstack(sample_values[self.labels_names_stacked].values), 0, 1)
                else:
                    y = np.vstack(sample_values[self.labels_names_stacked].values)[:,-1]
                    dev_y = dev_y[-1]

                yield sample_values.timestamp, img, y, dev_y
            else: # regular joint state - us2conf
                img = img / 255.0
                if self.cfg.data.sequence:
                    y = np.swapaxes(np.vstack(sample_values[self.labels_names_stacked].values), 0, 1)
                else:
                    y = np.vstack(sample_values[self.labels_names_stacked].values)[:,-1]
            
                yield sample_values.timestamp, img, y

    def test_generator(self):
        # yield image and label by iterating the data
        for sample_idx in range(len(self.test_df)):

            # draw sample
            sample_values = self.test_df.iloc[sample_idx]

            # skip if sample does not hold images from the same recording session
            if sample_values.sample_index < self.cfg.data.append * self.cfg.data.step - 1:
                continue

            if self.cfg.use_imgs: # need for images
                # append images
                imgs = []
                for j in range(len(sample_values['imgs'])):
                    # augment image using the random state and normalize
                    img = sample_values['imgs'][j]
                    # convert to RGB if asked
                    if self.cfg.data.channels == 3:
                        imgs.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                    imgs.append(np.expand_dims(img,axis=-1))
                img = np.concatenate(imgs, axis=-1)

            # yield inputs and labels for each mode
            if self.cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2multikey':
                img = img / 255.0
                dev_y = sample_values[self.device_label_name_stacked]
                if not self.cfg.data.sequence:
                    dev_y = dev_y[-1]
            
                yield img, dev_y
            elif self.cfg.mode == 'us2conf2multimidi' or self.cfg.mode == 'us2conf2multikey':
                img = img / 255.0
                dev_y = sample_values[self.device_label_name_stacked]
                if self.cfg.data.sequence:
                    y = np.swapaxes(np.vstack(sample_values[self.labels_names_stacked].values), 0, 1)
                else:
                    y = np.vstack(sample_values[self.labels_names_stacked].values)[:,-1]
                    dev_y = dev_y[-1]

                yield sample_values.timestamp, img, y, dev_y
            else: # regular joint state - us2conf
                img = img / 255.0
                if self.cfg.data.sequence:
                    y = np.swapaxes(np.vstack(self.test_df[self.labels_names_stacked].iloc[sample_idx].values), 0, 1)
                else:
                    y = np.vstack(self.test_df[self.labels_names_stacked].iloc[sample_idx].values)[:,-1]
            
                yield sample_values.timestamp, img, y

###################################################
#                 Augmentations                   #
###################################################

    def random_augment(self, x, rand_state):

        x = self.flip_horizontal(x, rand_state) # flip horizontally
        #x = self.flip_vertical(x, rand_state) # flip vertically
        x = self.random_brightness(x, rand_state) # change brightness randomly
        x = self.shift_horizontal(x, rand_state) # shift horizontally
        #x = self.random_rotate90(x, rand_state) # rotate by multiplying 90 degrees
        x = self.random_crop_resize(x, rand_state) # random crop and resize

        return x

    # flip horizontally
    def flip_horizontal(self, x, rand_state):
        if rand_state[0][0] == 1:
            if isinstance(x, np.ndarray):
                return np.fliplr(x)
            else:
                return ImageOps.mirror(x)
        else:
            return x
            
    # flip vertically
    def flip_vertical(self, x, rand_state):
        if rand_state[0][1] == 1:
            if isinstance(x, np.ndarray):
                return np.flipud(x)
            else:
                return ImageOps.flip(x)
        else:
            return x
    
    # changing brightness randomly
    def random_brightness(self, x, rand_state):
        if isinstance(x, np.ndarray):
            x = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
            x[:, :, 2] = np.clip(x[:, :, 2] * rand_state[1], 0, 255)
            return cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        else:
            enhancer = ImageEnhance.Brightness(x)
            return enhancer.enhance(rand_state[1]) 

    # shift horizontally and fill with noise
    def shift_horizontal(self, x, rand_state):
        if rand_state[2] != 0:
            if isinstance(x, np.ndarray):
                shifted_x = np.zeros_like(x)
                if rand_state[2] < 0:
                    shifted_x[:,-rand_state[2]:] = x[:,:rand_state[2]]
                else:
                    shifted_x[:,:-rand_state[2]] = x[:,rand_state[2]:]
                return shifted_x
            else:
                return x.transform(x.size, Image.AFFINE, (1, 0, rand_state[2], 0, 1, 0), resample=Image.BICUBIC)
        else:
            return x

    # rotate by multiplying 90 degrees
    def random_rotate90(self, x, rand_state):
        if rand_state[3] > 0:
            if isinstance(x, np.ndarray):
                return np.rot90(x, k=rand_state[3])
            else:
                return x.rotate(angle=rand_state[3] * 90, resample=Image.BICUBIC)
        else:
            return x
            
    # random crop and resize
    def random_crop_resize(self, x, rand_state):

        # get cropping window
        rand_left = rand_state[4][0]
        rand_upper = rand_state[4][1]
        rand_right = self.cfg.data.res - rand_state[4][2]
        rand_lower = rand_upper + rand_right - rand_left

        if isinstance(x, Image.Image):
            x = x.crop(box=(rand_left, rand_upper, rand_right, rand_lower))
            return x.resize(size=(self.cfg.data.res,self.cfg.data.res))
        else: # np.ndarray
            x = x[rand_upper:rand_lower,rand_left:rand_right] 
            return cv2.resize(x, (self.cfg.data.res,self.cfg.data.res))