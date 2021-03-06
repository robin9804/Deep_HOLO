import tensorflow.keras as keras
from functools import partial
import tensorflow as tf

# 기본적인 컨볼루젼에 대한 정의를 해줌
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

# ResNet에 대한 클래스
class ResidualUnit(keras.layers.Layer):
    # 필터 : 추출하고자 하는 특징의 갯수
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        # 활성화 함수로 relu 사용
        self.activation = keras.activations.get(activation)
        self.main_layers=[
            DefaultConv2D(filters, strides=strides),    # 컨볼루션 층을 사용
            keras.layers.BatchNormalization(),          # 배치 노멀라이제이션
            self.activation,                            # 활성화 함수 적용
            DefaultConv2D(filters),                     # 다시 합성곱층
            keras.layers.BatchNormalization()]          # 배치 노멀라이제이션 적용
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [DefaultConv2D(filters, kernel_size=1, strides=strides),
                                keras.layers.BatchNormalization()]

    def call(self, inputs):
        # 번호에 따라 몇 번 스킵하는 메서드
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z+skip_Z)

class ResNet(keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.num_classes = 1000
        #self.block_1 = ResNetBlock()
        #self.block_2 = ResNetBlock()
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.classifier = keras.layers.Dense(self.num_classes)

    def call(self, inputs):
        x = ResNetBlock(inputs)
        x = ResNetBlock(x)
        x = self.global_pool(x)
        return self.classifier(x)

def ResNetBlock(inputs):
    x = tf.nn.conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME')
    x = tf.nn.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x, filters=64, padding='SAME', kernel_size=3, strides=1)
    x = tf.nn.batch_normalization(x)
    x += inputs
    out = tf.nn.relu(x)
    return out

inputs = keras.Input(shape=(28,28,1), name='img')

resnet = ResNet()
#resnet.call(inputs)
#resnet.summary()
print(resnet.get_config())