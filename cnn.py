'''
2022/6/13
cifar10Prj
cifar10.py
by yhlin
'''
from datetime import datetime

import keras
import matplotlib.pyplot as plt
from keras import Sequential, Input, Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.datasets import cifar10
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Add, \
    GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import to_categorical
import tensorflow as tf


class CNN:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    model: tf.keras.Model
    conv: Conv2D
    kernel_size: tuple
    w: int
    h: int

    # num_channel, width, height
    def __init__(self):
        (self.train_images, self.train_labels), \
        (self.test_images, self.test_labels) = cifar10.load_data()
        self.w = self.train_images.shape[1]
        self.h = self.train_images.shape[2]
        self.kernel_size = (3, 3)
        self.chanel = 3

    '''訓練樣本與測試樣本正規化'''

    def normalization(self):
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0

    # One_hot 編碼
    def label_preprocess(self):
        self.train_labels = to_categorical(self.train_labels, 10)
        self.test_labels = to_categorical(self.test_labels, 10)

    # CNN model
    def cnn_model(self) -> keras.Model:
        model = Sequential()
        # 第一個卷積塊 Conv->Conv->Pool->Dropout
        model.add(Conv2D(32, self.kernel_size, activation='relu'
                         , padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(self.w, self.kernel_size, activation='relu'
                         , padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # 第二個卷積塊 Conv->Conv->Pool->Dropout
        model.add(Conv2D(64, self.kernel_size, activation='relu'
                         , padding='same'))
        model.add(Conv2D(64, self.kernel_size, activation='relu'
                         , padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        # 編譯模型
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc', 'mae'])
        return model

    def train(self, model: tf.keras.Model):
        start_time = datetime.now()
        early_stop = EarlyStopping(monitor='val_loss', patience=20)
        history = model.fit(self.train_images, self.train_labels, batch_size=128
                            , epochs=500, validation_split=0.2
                            )
        fig, (ax1, ax2) = plt.subplots(2, 1)
        end_time = datetime.now()
        total_time = end_time - start_time
        (test_loss, test_acc, test_mae) = model.evaluate(self.test_images, self.test_labels)
        print(
            'loss:{:.3f}\n acc:{:.3f}\n mae:{:.3f}\n  total time:{}'.format(test_loss, test_acc, test_mae, total_time))

        ax1.plot(history.history['acc'], label='acc')
        ax1.plot(history.history['val_acc'], label='val_acc')
        # ax1.legend(loc='best')

        ax2.plot(history.history['mae'], label='mae')
        ax2.plot(history.history['val_mae'], label='val_mae')
        # ax2.legend(loc='best')

        plt.show()

    def draw_cifar(self):
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(self.train_images[i])
        plt.show()


if __name__ == '__main__':
    analysis = CNN()
    analysis.normalization()
    analysis.label_preprocess()
    model = analysis.cnn_model()
    analysis.train(model)
