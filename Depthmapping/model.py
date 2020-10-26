import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers


input_tensor = tf.keras.Input(shape=(1024,1024,3), dtype='float32', name='input')

def unet(inputs):
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.UpSampling2D(size=(2,2))(drop5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = layers.Concatenate(axis=3)([drop4, up6])
    conv6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = layers.Concatenate(axis=3)([conv3, up7])
    conv7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = layers.Concatenate(axis=3)([conv2, up8])
    conv8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = layers.Concatenate(axis=3)([conv1, up9])

    conv9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    model = tf.keras.Model(inputs, conv10)

    #model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

"""

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input (InputLayer)              [(None, 1024, 1024,  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 1024, 1024, 6 1792        input[0][0]                      
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 1024, 1024, 6 36928       conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 512, 512, 64) 0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 512, 512, 128 73856       max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 512, 512, 128 147584      conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 256, 256, 128 0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 256, 256, 256 295168      max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 256, 256, 256 590080      conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 128, 128, 256 0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 128, 128, 512 1180160     max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 128, 128, 512 2359808     conv2d_6[0][0]                   
__________________________________________________________________________________________________
dropout (Dropout)               (None, 128, 128, 512 0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 64, 64, 512)  0           dropout[0][0]                    
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 64, 64, 1024) 4719616     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 64, 64, 1024) 9438208     conv2d_8[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 64, 64, 1024) 0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 128, 128, 102 0           dropout_1[0][0]                  
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 128, 128, 512 2097664     up_sampling2d[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 128, 128, 102 0           dropout[0][0]                    
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 128, 128, 512 2097664     concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 128, 128, 512 1049088     conv2d_11[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 256, 256, 512 0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 256, 256, 256 524544      up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 256, 256, 512 0           conv2d_5[0][0]                   
                                                                 conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 256, 256, 256 524544      concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 256, 256, 256 262400      conv2d_14[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 512, 512, 256 0           conv2d_15[0][0]                  
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 512, 512, 128 131200      up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 512, 512, 256 0           conv2d_3[0][0]                   
                                                                 conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 512, 512, 128 131200      concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 512, 512, 128 65664       conv2d_17[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 1024, 1024, 1 0           conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 1024, 1024, 6 32832       up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 1024, 1024, 1 0           conv2d_1[0][0]                   
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 1024, 1024, 6 32832       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 1024, 1024, 6 16448       conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 1024, 1024, 2 1154        conv2d_21[0][0]                  
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 1024, 1024, 1 3           conv2d_22[0][0]                  
==================================================================================================
Total params: 25,810,437
Trainable params: 25,810,437
Non-trainable params: 0
__________________________________________________________________________________________________

Process finished with exit code 0

"""



if __name__ == '__main__':
    unet(input_tensor)


