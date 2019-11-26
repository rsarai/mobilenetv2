"""
Train the MobileNet V2 model
"""
import os
import sys
import time
import argparse

import pandas as pd

from mobilenet_v2 import MobileNetv2

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
from keras.datasets import mnist


def get_mnist_dataset():
    #download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #reshape data to fit model
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    from keras.utils import to_categorical
    # convert class vectors to binary class matrices (https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/examples/mnist_cnn.py#L43)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def fine_tune(num_classes, weights, model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights(weights)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model


def train(batch, epochs, num_classes, size, weights, tclasses):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
    """

    train_generator, validation_generator = get_mnist_dataset()
    count1, count2 = len(train_generator[0]), len(validation_generator[0])

    if weights:
        model = MobileNetv2((size, size, 1), tclasses)
        model = fine_tune(num_classes, weights, model)
    else:
        model = MobileNetv2((size, size, 1), num_classes)

    opt = RMSprop()
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(
        train_generator[0], train_generator[1],
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        shuffle=True,
        callbacks=[earlystop])

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('model/history.csv', encoding='utf-8', index=False)
    model.save_weights('model/weights.h5')
    return history
