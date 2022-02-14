import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, GRU, Flatten, MaxPooling2D, MaxPool3D, Dropout, Concatenate


class MultiFrameModel(Model):
    def __init__(self, cfg):
        super(MultiFrameModel, self).__init__()

        # get number of outputs according to mode name
        self.cfg = cfg
        self.num_inputs = self.cfg.data.append
        self.num_outputs = get_num_of_joints(self.cfg)
        self.n_filters = self.cfg.model.n_filters
        
        if self.cfg.model.backbone == 'tcn':
            self.build_tcn()
        else: # self.cfg.model.backbone == 'c3de'
            self.build_c3d()

        if self.cfg.mode == 'us2conf2multikey' or self.cfg.mode == 'us2conf2multimidi':
            self.build_dev_decoder()

    # build C3D
    # https://arxiv.org/pdf/1412.0767.pdf
    def build_c3d(self):

        # input stream
        self.c_conv1 = Conv3D(filters=self.n_filters*2, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
        self.c_conv2 = Conv3D(filters=self.n_filters*4, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
        self.c_conv3 = Conv3D(filters=self.n_filters*8, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
        self.c_conv4 = Conv3D(filters=self.n_filters*8, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
        if self.cfg.model.backbone == 'c3de':
            self.c_conv5 = Conv3D(filters=self.n_filters*16, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
            self.c_conv6 = Conv3D(filters=self.n_filters*16, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
            self.c_conv7 = Conv3D(filters=self.n_filters*16, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
            self.c_conv8 = Conv3D(filters=self.n_filters*16, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
            self.c_d1 = Dense(units=2048, activation='relu')
            #self.c_d2 = Dense(units=512, activation='relu')
        else: # self.cfg.model.backbone == 'c3d'
            self.c_conv5 = Conv3D(filters=self.n_filters*8, kernel_size=(3, 3, 3), strides=1, padding='same', activation='relu')
            self.c_d1 = Dense(units=2048, activation='relu')
            #self.c_d2 = Dense(units=2048, activation='relu')

        if self.cfg.mode == 'us2conf':
            self.c_d2 = Dense(units=self.num_outputs, activation='linear')
        else: # self.cfg.mode == 'us2multimidi' or self.cfg.mode == 'us2multikey'
            self.c_d2 = Dense(units=self.cfg.data.dev_classes, activation='linear')

        # pooling layers
        self.c_pool1 = MaxPool3D(pool_size=(2, 2, 1))
        self.c_pool2 = MaxPool3D(pool_size=(2, 2, 2))
        self.c_pool3 = MaxPool3D(pool_size=(2, 2, 2))
        self.c_pool4 = MaxPool3D(pool_size=(2, 2, 2))
        self.c_pool5 = MaxPool3D(pool_size=(2, 2, 2))

        # dense block
        self.c_flatten = Flatten()
        self.c_drop1 = Dropout(rate=0.5)
        self.c_drop2 = Dropout(rate=0.5)
        self.c_drop3 = Dropout(rate=0.5)
        self.c_drop4 = Dropout(rate=0.5)
        self.c_drop5 = Dropout(rate=0.5)
        #self.c_d3 = Dense(units=self.cfg.model.z_size, activation='linear')

    def build_tcn(self):

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
        self.flatten_tcn = Flatten()
        self.merge_tcn = Concatenate(axis=1)

        self.tcn_gru1 = GRU(units=1024, activation='relu', return_sequences=True)
        self.tcn_gru2 = GRU(units=128, activation='relu', return_sequences=True)

        if self.cfg.data.sequence or self.cfg.mode == 'us2conf2multikey' or self.cfg.mode == 'us2conf2multimidi':
            if self.cfg.mode == 'us2conf' or self.cfg.mode == 'us2conf2multikey' or self.cfg.mode == 'us2conf2multimidi':
                self.tcn_dgru = GRU(units=self.num_outputs, activation='linear', return_sequences=True)
            else:
                self.tcn_dgru = GRU(units=self.cfg.data.dev_classes, activation='linear', return_sequences=True)
        else:
            if self.cfg.mode == 'us2conf':
                self.tcn_dgru = Dense(units=self.num_outputs, activation='linear')
            else:
                self.tcn_dgru = Dense(units=self.cfg.data.dev_classes, activation='linear')

        #self.tcn_dgru = Dense(units=self.cfg.model.z_size*2)
        #self.dm_gru3 = GRU(units=self.cfg.data.dev_classes, activation='linear', return_sequences=True) # softmax
        # self.dc_gru3 = GRU(units=self.num_outputs, activation='linear', return_sequences=True)

    def build_dev_decoder(self):

        self.merge_dev = Concatenate(axis=-1)

        self.dev_gru1 = GRU(units=256, activation='relu', return_sequences=True)
        self.dev_gru2 = GRU(units=128, activation='relu', return_sequences=True)
        if self.cfg.data.sequence:
            self.dev_dgru = GRU(units=self.cfg.data.dev_classes, activation='linear', return_sequences=True)
        else:
            self.dev_dgru = Dense(units=self.cfg.data.dev_classes, activation='linear')

    def call(self, x):

        # encode nodes into a preliminary latent vector
        if self.cfg.model.backbone == 'tcn':
            ze, ye = self.encode_imgs_tcn(x)
        else: # self.cfg.model.backbone == 'c3de'
            ye = self.encode_imgs_c3d(x)
        
        if self.cfg.mode == 'us2conf2multikey' or self.cfg.mode == 'us2conf2multimidi':

            if self.cfg.model.res_layer:
                yc = self.merge_dev([ze, ye])
                return ye, self.decode_dev(yc)
            else:
                return ye, self.decode_dev(ye)
        else:
            return ye

    def encode_imgs_c3d(self, x):

        # convert input from NHWC to NHWLC
        x = tf.expand_dims(x, axis=-1)

        # input stream
        x = self.c_pool1(self.c_conv1(x))
        x = self.c_drop1(self.c_pool2(self.c_conv2(x)))
        if self.cfg.model.backbone == 'c3de':
            x = self.c_drop2(self.c_pool3(self.c_conv4(self.c_conv3(x))))
            x = self.c_drop3(self.c_pool4(self.c_conv6(self.c_conv5(x))))
            x = self.c_pool5(self.c_conv8(self.c_conv7(x)))
        else: # self.cfg.model.backbone == 'c3d':
            x = self.c_drop2(self.c_pool3(self.c_conv3(x)))
            x = self.c_drop3(self.c_pool4(self.c_conv4(x)))
            x = self.c_pool5(self.c_conv5(x))

        # dense block
        x = self.c_flatten(x)
        x = self.c_drop4(self.c_d1(x))
        #x = self.c_drop5(self.c_d2(x))
        x = self.c_d2(x)
        #return self.c_d3(x)
        return x

    # forward function to encode using UNet encoder
    def encode_img_unet(self, x):

        # 1st conv block
        x = self.tcn_pool1(self.tcn_conv12(self.tcn_conv11(x)))
        x = self.tcn_pool2(self.tcn_conv22(self.tcn_conv21(x)))
        x = self.tcn_pool3(self.tcn_conv32(self.tcn_conv31(x)))
        x = self.tcn_pool4(self.tcn_drop4(self.tcn_conv42(self.tcn_conv41(x))))
        return self.tcn_drop5(self.tcn_conv52(self.tcn_conv51(x)))

    def encode_imgs_tcn(self, x):

        # separate X space
        xs = tf.split(x, self.num_inputs, axis=-1)

        zs = []
        for x in xs:
            x_features = self.encode_img_unet(x)
            features = tf.expand_dims(self.flatten_tcn(self.pool_tcn(x_features)), axis=1)
            zs.append(features)

        z = self.merge_tcn(zs)

        z = self.tcn_gru1(z)
        z1 = self.tcn_gru2(z)
        if not self.cfg.data.sequence:
            return self.tcn_dgru(self.flatten_tcn(z1))
        else: 
            return z1, self.tcn_dgru(z1)

    def decode_dev(self, y):

        y = self.dev_gru1(y)
        y = self.dev_gru2(y)
        if not self.cfg.data.sequence:
            return self.dev_dgru(self.flatten_tcn(y))
        else: 
            return self.dev_dgru(y)

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


# utility function to return number of output variables given cfg
def get_num_of_joints(cfg):

    if cfg.data.joints_version == '1':
        return 11
    elif cfg.data.joints_version == '1s':
        return 4
    elif cfg.data.joints_version == '2':
        return 15
    elif cfg.data.joints_version == '2c':
        return 6
    elif cfg.data.joints_version == '3':
        return 15
    elif cfg.data.joints_version == '4':
        return 17
    elif cfg.data.joints_version == 'custom':
        return len(cfg.data.joint_names)
    else: # mode == 'plain'
        return 0