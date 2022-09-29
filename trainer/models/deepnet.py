
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout, BatchNormalization
from trainer.models.utils import get_num_of_joints


# model definition class
class DeepNetModel(Model):
    def __init__(self, cfg):
        super(DeepNetModel, self).__init__()

        # get number of outputs according to mode name
        self.cfg = cfg
        self.num_outputs = get_num_of_joints(self.cfg)

        # create backbone
        if self.cfg.model.backbone == 'mobilenet':
            self.backbone = MobileNetV2(include_top=False)
        elif self.cfg.model.backbone == 'unet':
            self.build_unet_encoder()
        else: 
            self.backbone = VGG16(include_top=False)

        self.flatten = Flatten()
        self.dropout = Dropout(rate=0.5)
        self.d1 = Dense(units=1024, activation='relu')
        if self.cfg.mode == 'us2conf':
            self.d2 = Dense(units=self.num_outputs, activation='linear')
        else: # cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2multikey'
            self.d2 = Dense(units=self.cfg.data.dev_classes, activation='linear')

    def build_unet_encoder(self):

        # encoding part
        self.tcn_conv11 = Conv2D(self.cfg.model.n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_conv12 = Conv2D(self.cfg.model.n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.tcn_conv21 = Conv2D(self.cfg.model.n_filters * 2,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_conv22 = Conv2D(self.cfg.model.n_filters * 2,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.tcn_conv31 = Conv2D(self.cfg.model.n_filters * 4,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_conv32 = Conv2D(self.cfg.model.n_filters * 4,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_pool3 = MaxPooling2D(pool_size=(2, 2))

        self.tcn_conv41 = Conv2D(self.cfg.model.n_filters * 8,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_conv42 = Conv2D(self.cfg.model.n_filters * 8,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_drop4 = Dropout(0.3)
        self.tcn_pool4 = MaxPooling2D(pool_size=(2, 2))

        self.tcn_conv51 = Conv2D(self.cfg.model.n_filters * 16,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_conv52 = Conv2D(self.cfg.model.n_filters * 16,3,activation='relu',padding='same',kernel_initializer='he_normal')
        self.tcn_drop5 = Dropout(0.3)

        # concatenate images into a temporal input
        self.pool_tcn = tf.keras.layers.MaxPool2D(pool_size=(4,4))

    def call(self, x):
        if self.cfg.model.backbone == 'unet':
            x = self.encode_img_unet(x)
        else:
            x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.d1(x)
        return self.d2(x)

    # forward function to encode using UNet encoder
    def encode_img_unet(self, x):

        x = self.tcn_pool1(self.tcn_conv12(self.tcn_conv11(x)))
        x = self.tcn_pool2(self.tcn_conv22(self.tcn_conv21(x)))
        x = self.tcn_pool3(self.tcn_conv32(self.tcn_conv31(x)))
        x = self.tcn_pool4(self.tcn_drop4(self.tcn_conv42(self.tcn_conv41(x))))
        x = self.tcn_drop5(self.tcn_conv52(self.tcn_conv51(x)))
        return self.pool_tcn(x)
