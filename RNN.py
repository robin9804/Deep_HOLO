import tensorflow as tf
import numpy as np

# 순환 신경망을 케라스로 구현합니다.

model = tf.keras.Sequential()

# embedding layer를 구현합니다.
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))

# LSTM 레이어를 구현합니다.
model.add(tf.keras.layers.LSTM(128))  # units 128개

# 완전연결층을 구현합니다.
model.add(tf.keras.layers.Dense(10))

model.summary()

"""
결과 Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          64000     
_________________________________________________________________
lstm (LSTM)                  (None, 128)               98816     
_________________________________________________________________
dense (Dense)                (None, 10)                1290      
=================================================================
Total params: 164,106
Trainable params: 164,106
Non-trainable params: 0
_________________________________________________________________
"""