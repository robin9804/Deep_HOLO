from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from Resnet import ResidualUnit

# 필터 10개짜리 ResNet을 만듭니다.
resnet = tf.keras.Sequential()
RES = ResidualUnit(10)
resnet.add(RES.main_layers)
#resnet.add(RES.skip_layers(1))
#resnet.add(RES.main_layers)
#resnet.add(RES.skip_layers(2))
resnet.add(Flatten())
resnet.add(Dense(100, activation='relu'))
resnet.add(Dense(10, activation='softmax'))
resnet.summary()