import tensorflow as tf
from tensorflow.keras import Input

# import data


def conv2d(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x


# input 정해주기
input_tensor = Input(shape=(224,224,3), dtype='float32', name='input')
x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, activation='relu')(input_tensor)
x = tf.keras.layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
x = conv2d(x, 64)
x = conv2d(x, 128)
x = conv2d(x, 256)
x = conv2d(x, 512)
#x = tf.keras.layers.Flatten()(x)  얘가 차원을 줄여벌였다.
x = tf.keras.layers.GlobalAveragePooling2D()(x)
res34 = tf.keras.layers.Dense(1000, activation='softmax')(x)

model = tf.keras.Model(input_tensor, res34)
model.summary()
"""
결과
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           [(None, 224, 224, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 109, 109, 64)      9472      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 55, 55, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 55, 55, 64)        36928     
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 55, 55, 64)        36928     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 55, 55, 64)        36928     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 55, 55, 64)        36928     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 55, 55, 128)       73856     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 55, 55, 128)       147584    
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 55, 55, 128)       147584    
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 55, 55, 128)       147584    
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 55, 55, 256)       295168    
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 55, 55, 256)       590080    
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 55, 55, 256)       590080    
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 55, 55, 256)       590080    
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 55, 55, 512)       1180160   
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 55, 55, 512)       2359808   
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 55, 55, 512)       2359808   
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 55, 55, 512)       2359808   
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
=================================================================
Total params: 10,998,784
Trainable params: 10,998,784
Non-trainable params: 0
_________________________________________________________________
"""
