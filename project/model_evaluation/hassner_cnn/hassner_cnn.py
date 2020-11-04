'''
Defines factory method for generating Hassner's CNN.
reference: https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf
'''
import tensorflow as tf


def get_hassner_cnn(input_shape: tuple, num_classes: int) -> tf.keras.Sequential:
    '''
    Creates Hassner's CNN. Needs to be compiled before using.

    :params:
    - input_shape (int): size of input images - tuple of width and height
    - num_classes (int): number of classes to predict

    :return:
    - (tf.keras.Sequential)
    '''
    pool_size = (3, 3)
    strides = (2, 2)

    # custom layer
    input0 = tf.keras.layers.InputLayer(
        input_shape=input_shape,
    )
    
    conv1 = tf.keras.layers.Conv2D(
        filters=96,
        kernel_size=7,
        activation='relu',
    )

    pool1 = tf.keras.layers.MaxPool2D(
        pool_size=pool_size,
        strides=strides,
    )

    norm1 = tf.keras.layers.Lambda(
        tf.nn.local_response_normalization,
    )

    conv2 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=5,
        activation='relu',
    )

    pool2 = tf.keras.layers.MaxPool2D(
        pool_size=pool_size,
        strides=strides,
    )

    norm2 = tf.keras.layers.Lambda(
        tf.nn.local_response_normalization,
    )

    conv3 = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=3,
        activation='relu',
    )

    pool5 = tf.keras.layers.MaxPool2D(
        pool_size=pool_size,
        strides=strides,
    )
    
    # custom layer
    flatten5 = tf.keras.layers.Flatten()

    fc6 = tf.keras.layers.Dense(
        units=512,
        activation='relu',
    )

    drop6 = tf.keras.layers.Dropout(rate=0.5)

    fc7 = tf.keras.layers.Dense(
        units=512,
        activation='relu',
    )

    drop7 = tf.keras.layers.Dropout(rate=0.5)

    fc8 = tf.keras.layers.Dense(
        units=num_classes,
        activation='softmax',
    )

    return tf.keras.Sequential([
        input0,
        conv1,
        pool1,
        norm1,
        conv2,
        pool2,
        norm2,
        conv3,
        pool5,
        flatten5,
        fc6,
        drop6,
        fc7,
        drop7,
        fc8,
    ])
