import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 로드
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()
# 데이터의 리쉐잎
train_img = train_img.reshape((60000, 28, 28,1))    # 6만개 샘플, 28 x 28 크기, mono8
test_img = test_img.reshape((10000, 28,28,1))       # test 셋은 1만개
# 픽셀 값을 0과 1사이로 정규화
train_img, test_img = train_img / 255.0, test_img / 255.0

def resnet(x):
    inputs = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation=None)(x)
    return x + inputs

input_tensor = tf.keras.Input(shape=(28,28,1), dtype='float32', name='Resnet')
x = resnet(input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
res34 = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(input_tensor, res34)
# model.summary()

# loss를 sparse 카테고리컬 크로스 엔트로피로 바꿈
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
his = model.fit(train_img, train_labels, epochs=10)


plt.plot(his.history['accuracy'])
plt.show()

# 모델 저장하는 부분
model.save('model.h5')
# 아키텍처 따로 json, 등을 저장
json_str = model.to_json()
# weight 따로 저장
# model.save_weights( '경로를 넣어 주세요' )