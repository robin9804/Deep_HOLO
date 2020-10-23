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



import matplotlib.pyplot as plt
import numpy as np
from ophpy import Depthmap, Image2D, PointCloud


def test(mode):
    if mode == 'PointCloud':
        # 1. Point Cloud base CGH generation
        input_data = 'PointCloud_Dice_RGB.ply'
        RS = PointCloud.Propagation(input_data, method='RS', angleY=1)
        Red_image = RS.CalHolo('red')  # RS integral methods
        plt.imshow(np.angle(Red_image))  # show phase angle data of red light fringe pattern
        RS.getRGBImage(Red_image, RS.CalHolo('green'), RS.CalHolo('blue'), 'test file name.bmp', type='angle')

    elif mode == '2Dimage':
        # 2. 2D image base CGH generation
        input_img = 'Dice_RGB.bmp'
        f = Image2D.Propagation(input_img, angleY=0.8)
        Red_image = f.Fresnel('red')  # Fresnel propagation using Single FFT
        plt.imshow(np.angle(Red_image))
        f.getRGBImage(Red_image, f.Fresnel('green'), f.Fresnel('blue'), 'test file name.bmp', type='angle')

    elif mode == 'Depthmap':
        # 3. Depthmap base CGH generation
        input_img = 'Dice_RGB.bmp'
        input_depthmap = 'Dice_depth.bmp'
        D = Depthmap.Propagation(input_img, input_depthmap)
        Red_image = D.parallelCal('red')  # using parallel calculation
        plt.imshow(np.angle(Red_image))
        D.getRGBImage(Red_image, D.parallelCal('green'), D.parallelCal('blue'), 'test file name.bmp', type='angle')


if __name__ == '__main__':
    test('2Dimage')  # enter type of source



