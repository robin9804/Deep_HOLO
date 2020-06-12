import tensorflow as tf
import numpy as np

# Res unit 만들기
class ResUnit(tf.keras.models):
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME'):
        super(ResUnit, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding=padding,
                                           kernel_initializer='random_normal')
        self.BN = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        """레이어를 부르는 메소드"""
        layer = tf.nn.max_pool2d(inputs, (2, 2))
        layer = self.conv(layer)
        layer = self.BN(layer)
        layer = tf.nn.relu(layer)  # 활성화 함수 지정
        layer = self.conv(layer)
        layer = self.BN(layer)
        layer = tf.nn.relu(layer)  # 활성화 함수 지정
        layer = layer + inputs
        return layer

class SubpixelConv(tf.keras.models):
    def __init__(self, filters=64, kernel_size=3, strides=1, padding='SAME'):
        super(SubpixelConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding=padding,
                                           kernel_initializer='random_normal')
        self.BN = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        layer = self.conv(inputs)


