import numpy as np
import tensorflow as tf
from PIL import Image


# CH 추출
def chGen_mono(raw, w=2448, h=2048):
    """Get mono Complex image"""
    with open(raw, 'r') as rawimg:
        img = np.fromfile(rawimg, np.dtype('u1'), w * h).reshape(h, w)
        img = img[:, (w - h) // 2:(w + h) // 2]
        img = np.reshape(img, [1, 2048, 2048, 1])
    # convolution
    k1 = tf.constant([[1, 0], [0, -1]])
    k1 = tf.reshape(k1, [2, 2, 1, 1])
    k2 = tf.constant([[0, 1], [-1, 0]])
    k2 = tf.reshape(k2, [2, 2, 1, 1])
    # real part
    img_r = tf.nn.conv2d(img, k1, strides=[1, 2, 2, 1], padding="VALID")
    img_r = np.reshape(img_r, [1024, 1024])
    # imag part
    img_i = tf.nn.conv2d(img, k2, strides=[1, 2, 2, 1], padding="VALID")
    img_i = np.reshape(img_i, [1024, 1024])
    # complex img
    ch = img_r + 1j * img_i
    return ch


def chGen_color(raw, w=2448, h=2048):
    with open(raw, 'rb') as f:
        img = Image.fromstring('RGB', )
    return img
