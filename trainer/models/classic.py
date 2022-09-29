from turtle import xcor
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, GRU
from tensorflow.keras.layers import Flatten, AveragePooling2D, Reshape
from trainer.models.utils import get_num_of_joints


class EchoFlexDownsampler(Model):
    def __init__(self, cfg):
        super(EchoFlexDownsampler, self).__init__()

        self.cfg = cfg
        self.pool_size = 20
        self.features_size = (self.cfg.data.res // self.pool_size) ** 2

        self.avgpool = AveragePooling2D(pool_size=(self.pool_size,self.pool_size))
        self.reshape = Reshape([self.features_size])

    def call(self, x):

        x = self.avgpool(x)
        x = tf.reduce_sum(x, axis=-1)

        return self.reshape(x)


class PerforDownsampler(Model):
    def __init__(self, cfg):
        super(PerforDownsampler, self).__init__()

        self.cfg = cfg
        self.num_lines = 5
        self.gather_step = self.cfg.data.res // (self.num_lines + 1)
        self.window_size = 20

        self.reshape1 = Reshape([self.window_size, -1, self.cfg.data.append])
        self.reshape2 = Reshape([-1, self.window_size])


    def call(self, x):

        # gather 5 columns uniformly
        x = x[:,:,::self.gather_step,:]
        x = x[:,:-4,1:-1,:]

        # divide into windows
        x = self.reshape1(x)

        x = tf.transpose(x, [0,2,3,1])
        x = self.reshape2(x)

        return x
