from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation
import tensorflow as tf
from Resnet import ResidualUnit

# keras 모듈에 있는 RESNET50을 사용하기
#model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

resnet = tf.keras.applications.resnet.ResNet50(include_top=False,   # 네트워크 최상단에 완전연결 레이어 넣기
                                               weight=None,         # 임의의 초기값=None, 혹은 Imagenet(선행학습)
                                               input_tensor=None,   # 모델 이미지의 인풋으로 사용할 수 있는 선택적 케라스 텐서
                                               input_shape=(224,224,3),  # 선택적 형태의 튜플,
                                               pooling='max')

class Model(tf.keras.models):
    def __init__(self, label_dim):
        super(Model, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal()  # 가중치 초기화
        self.model = tf.keras.Sequential()
        self.model.add(Flatten())

        for i in range(2):
            self.model.add(Dense(256, use_bias=True, kernel_initializer=weight_init, activation='sigmoid'))
        self.model.add(Dense(label_dim, use_bias=True, kernel_initializer=weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x

# 하이퍼 파라미터 결정
batch_size = 128

training_epochs = 1
# training_iterations = len(train_x) // batch_size

label_dim = 10

train_flag = True

# define model
networt = Model(label_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

networt.summary()
