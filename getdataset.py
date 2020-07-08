import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# parameters
w = 2448
h = 2048
num_file = 5

out_mat = np.zeros((num_file, 1024, 1024, 2))
h5name = 'dataset.h5'

# hdf5 file open
f = h5py.File('data4.h5', 'w')

for i in range(num_file):
    # num = "d%d" %(i+1)
    fname = "test%d.raw" % (i + 1)
    with open(fname, 'r') as raw:
        img = np.fromfile(raw, np.dtype('u1'), w * h).reshape(h, w)
        img = img[:, (w - h) // 2: (w + h) // 2]  # crop
        img = np.reshape(img, [1, 2048, 2048, 1])

    # conv
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

    # fourier transform
    fftimg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ch)))

    # output matrix
    outimg = np.zeros((1024, 1024, 2))
    outimg[:, :, 0] = np.real(fftimg)
    outimg[:, :, 1] = np.imag(fftimg)

    out_mat[i, :, :, :] = outimg

f.create_dataset('data', data=out_mat)

f.close()

# read data
x_data = tf.keras.utils.HDF5Matrix('data4.h5', 'data')
print(x_data.shape)