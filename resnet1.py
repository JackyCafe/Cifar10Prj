from keras import Input, Model
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Add, Conv2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

kernel_size =3
def res_net_model() -> keras.Model:
    input = Input(shape=(32, 32, 3))
    x = conv(16, 3)(input)  # 第一層
    x = residual_block(64, 1, 18)(x)
    x = residual_block(128, 2, 18)(x)
    x = residual_block(256, 2, 18)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    model = Model(inputs=input, output=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD('momentum=0.9'),
                  metrics=['acc', 'mae'])
    return model


def data_generator():
    train_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True
    )
    test_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    for data in (train_gen, test_gen):
        data.fit(train_images)


def step_decay(epoch):
    x = 0.1
    if epoch >= 80: x = 0.01
    if epoch >= 120: x = 0.001
    return x

    # 殘差塊A


def first_residual_unit(filters, strides):
    def f(x):
        # ->BN->Relu
        x = BatchNormalization()(x)
        x_b = Activation('relu')(x)

        x = conv(filters // 4, 1, strides=strides)(x_b)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(filters // 4, 3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(filters, 1)(x)
        x_b = conv(filters, 1, strides)(x_b)
        return Add()[x, x_b]

    return f

    # 殘差塊B


def residual_unit(filters):
    def f(x):
        x_b = x
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(filters // 4, 1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(filters // 4, 3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(filters, 1)(x)
        return Add()[x, x_b]

    return f

    # 殘差塊Ax1 + BX17


def residual_block(filters, strides, unit_size):
    def f(x):
        x = first_residual_unit(filters, strides)(x)
        for i in range(unit_size - 1):
            x = residual_unit(filters)(x)
        return x

    return f

    # 建構捲積層


def conv( filters, strides=1):
    return Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.001))


if __name__ == '__main__':
    (train_images, train_labels), \
    (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)