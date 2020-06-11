from tensorflow.keras.layers import Dense, Flatten, Activation
import tensorflow as tf
from Resnet import ResidualUnit


# keras 모듈에 있는 RESNET50을 사용하기
#model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

Res = tf.keras.Sequential([ResidualUnit(10), Flatten(), Dense(100, activation='relu'),
                           Dense(10, activation='softmax')])
Res.summary()