# package import
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

# MNIST dataset import
from sklearn.model_selection import train_test_split

# 데이터 전처리
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2)
# random state 에 대해서도 원한다면 할 수 있다.

# 라벨 target 데이터에 대한 인코딩
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

# x 데이터 전처리
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1,28, 28, 1)

# x 데이터 노말라이즈
x_train = x_train/255
x_val = x_val/255

# 합성곱층 쌓기
conv1 = tf.keras.Sequential()  # 컨볼루션을 위한 모델을 하나 쌓기
conv1.add(Conv2D(10, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))  # 합성곱층
conv1.add(MaxPooling2D((2,2)))  # 2x2로 maxpooling
conv1.add(Flatten())  # 완전 연결층에 연결
conv1.add(Dense(100, activation='relu'))  # 완전연결층으로 하기
conv1.add(Dense(10, activation='softmax'))  # 완전연결층 10개로 소프트맥스

conv1.summary()