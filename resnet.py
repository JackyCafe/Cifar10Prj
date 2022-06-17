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
from keras_preprocessing.image import ImageDataGenerator


class Resnet:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    model: tf.keras.Model
    kernel_size: tuple
    w: int
    h: int

    # num_channel, width, height
    def __init__(self):
        (self.train_images, self.train_labels), \
        (self.test_images, self.test_labels) = cifar10.load_data()
        self.w = self.train_images.shape[1]
        self.h = self.train_images.shape[2]
        self.kernel_size = (3)
        self.chanel = 3

    '''訓練樣本與測試樣本正規化'''

    def normalization(self):
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0

    # One_hot 編碼
    def label_preprocess(self):
        self.train_labels = to_categorical(self.train_labels, 10)
        self.test_labels = to_categorical(self.test_labels, 10)

    # 建構模型
    def res_net_model(self) -> keras.Model:
        input = Input(shape=(32, 32, 3))
        x = self.conv(16, 3)(input)  # 第一層
        x = self.residual_block(64, 1, 18)(x)
        x = self.residual_block(128, 2, 18)(x)
        x = self.residual_block(128, 2, 18)(x)
        x = self.residual_block(256, 2, 18)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        output = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(x)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(momentum=0.9),
                      metrics=['acc', 'mae'])
        return model

    def data_generator(self):
        self.train_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True
        )
        self.test_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True
        )
        for data in (self.train_gen, self.test_gen):
            data.fit(self.train_images)

    def step_decay(self, epoch):
        x = 0.1
        if epoch >= 80: x = 0.01
        if epoch >= 120: x = 0.001
        return x

    # 殘差塊A
    def first_residual_unit(self, filters, strides):
        def f(x):
            # ->BN->Relu
            x = BatchNormalization()(x)
            x_b = Activation('relu')(x)
            x = self.conv(filters // 4, 1, strides=strides)(x_b)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = self.conv(filters // 4, 3)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = self.conv(filters, 1)(x)
            x_b = self.conv(filters, 1, strides)(x_b)
            return Add()([x, x_b])

        return f

    # 殘差塊B
    def residual_unit(self, filters):
        def f(x):
            x_b = x
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = self.conv(filters // 4, 1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = self.conv(filters // 4, 3)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = self.conv(filters, 1)(x)
            return Add()([x, x_b])

        return f

    # 殘差塊Ax1 + BX17
    def residual_block(self, filters, strides, unit_size):
        def f(x):
            x = self.first_residual_unit(filters, strides)(x)
            for i in range(unit_size - 1):
                x = self.residual_unit(filters)(x)
            return x

        return f

    # 建構捲積層
    def conv(self, filters,kernel_size, strides=1):
        return Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                      kernel_initializer='he_normal', kernel_regularizer=l2(0.001))

    def train(self, model: Model):
        batch_size = 128
        lr_decay = LearningRateScheduler(self.step_decay)
        early_stop = EarlyStopping(monitor='val_loss', patience=20)

        start_time = datetime.now()
        history = model.fit(
            self.train_gen.flow(self.train_images,self.train_labels,batch_size=batch_size)
            ,epochs=200,
            steps_per_epoch=self.train_images.shape[0]//batch_size,
            validation_data=self.test_gen.flow(self.test_images,self.test_labels,batch_size=batch_size),
            validation_steps=self.test_images.shape[0]//batch_size,
            callbacks=[lr_decay]
        )
        model.save('resnet_72.h5')
        fig, (ax1, ax2) = plt.subplots(2, 1)
        end_time = datetime.now()
        total_time = end_time - start_time
        (test_loss, test_acc, test_mae) = model.evaluate_generator(self.test_gen.flow(self.test_images, self.test_labels,batch_size=batch_size))
        # (test_loss, test_acc, test_mae) = model.evaluate(self.test_images, self.test_labels)
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
    resnet = Resnet()
    resnet.normalization()
    resnet.label_preprocess()
    resnet.data_generator()
    model = resnet.res_net_model()
    resnet.train(model)
