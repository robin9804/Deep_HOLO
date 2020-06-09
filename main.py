from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

import tensorflow as tf

model = tf.keras.models()

model.add(Dense(100,activation='relu'))
