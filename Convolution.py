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

# 라벨 target 데이터에 대한 인코딩,
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
# 특징 클래스 10개라서 10개의 필터 사용
# 10개를 쌓는다고 생각해도 무방.

conv1.add(MaxPooling2D((2,2)))              # 2x2로 maxpooling
conv1.add(Flatten())                        # 완전 연결층에 연결
conv1.add(Dropout(0.5))                     # 드롭아웃 삽입
conv1.add(Dense(100, activation='relu'))    # 100개의 유닛(뉴런)을 가진 완전연결층만들기
conv1.add(Dense(10, activation='softmax'))  # 완전연결층 10개로 소프트맥스

conv1.summary()                             # 모델에 대한 서머리

# 모델에 대한 옵티마이져와 손실함수 결정.
conv1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련하기
conv1.fit(x_train, y_train_encoded, epochs=20, validation_data=(x_val, y_val_encoded))

# 컨볼루션 레이어를 직접 구현한 경우
class Conv:
    def __init__(self, learning_rate=0.01, n_kernels=10, units=10, batch_size=32):
        self.lr = learning_rate         # 학습률
        self.n_kernels = n_kernels      # 커널 갯수, 여기서는 10개가 기본, 10개 클래스를 분류하기 때문
        self.units = units              # 분류를 목표로 하는 10개의 클래스
        self.batch_size = batch_size    # 미니배치의 사이즈 정해주기
        self.kernel_size = 3            # 커널의 크기,
        self.optimizer = None
        self.losses = []
        self.val_losses = []
        self.conv_w = None
        self.conv_b = None
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.a1 = None


    def forpass(self, x):
        # 3 x 3 합성곱 연산을 수행
        cout1 = tf.nn.conv2d(x, self.conv_w, strides=1, padding='SAME') + self.conv_b
        # relu 적용
        r_out = tf.nn.relu(cout1)
        # max pooling
        p_out = tf.nn.max_pool2d(r_out, ksize=2, strides=2, padding='VALID')
        # 출력을 일렬로 펼치기
        f_out = tf.reshape(p_out, [x.shape[0], -1])
        z1 = tf.matmul(f_out, self.w1) + self.b1  # 첫 번째 층의 선형식
        a1 = tf.nn.relu(z1)  # 활성화 함수(relu) 적용
        z2 = tf.matmul(a1, self.w2) + self.b2  # 두 번째 층
        return z2

    def training(self, x, y):
        m = len(x)
        with tf.GradientTape() as tape:
            z = self.forpass(x)  # 정방향 계산 수행
            loss = tf.nn.softmax_cross_entropy_with_logits(y, z)  # 손실계산
            loss = tf.reduce_mean(loss)
        weight_list = [self.conv_w, self.conv_b, self.w1, self.b1, self.w2, self.b2]
        grads = tape.gradient(loss, weight_list)
        self.optimizer.apply_gradients(zip(grads, weight_list))

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        self.init_weight(x.shape, y.shape[1])
        self.optimizer = tf.optimizers.SGD(learning_rate=self.lr)
        for i in range(epochs):
            print('에포크', i, end='')
            batch_losses = []
            for x_batch, y_batch in self.gen_batch(x, y):
                print('.', end='')
                self.training(x_batch, y_batch)
                batch_losses.append(self.get_loss(x_batch, y_batch))  # 배치 손실 기록
            print()
            self.losses.append(np.mean(batch_losses))
            self.val_losses.append(self.get_loss(x_val, y_val))

    def init_weight(self, input_shape, n_classes):
        g = tf.initializers.glorot_uniform()
        self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
        self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype=float)
        n_features = 14 * 14 * self.n_kernels
        self.w1 = tf.Variable(g((n_features, self.units)))  # 특성 크기와 은닉층 크기
        self.b1 = tf.Variable(np.zeros(self.units), dtype=float)
        self.w2 = tf.Variable(g((self.units, n_classes)))  # 은닉층의 크기, 클래스 갯수
        self.b2 = tf.Variable(np.zeros(n_classes), dtype=float)

    def gen_batch(self, x, y):
        bins = len(x) // self.batch_size  # 미니 배치 횟수
        indexes = np.random.permutation(np.arange(len(x)))  # 인덱스 섞기
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def predict(self, x):
        z = self.forpass(x)
        return np.argmax(z.numpy(), axis=1)

    def score(self, x, y):
        return np.mean(self.predict(x) == np.argmax(y, axis=1))

    def get_loss(self, x, y):
        z = self.forpass(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
        return loss.numpy()