import numpy as np
import os
# Utility for download digit dataset
from utils import (
    maybe_download,
    extract_data,
    extract_labels)
# Import keras modules
from keras.models import Sequential
from keras.layers.core import (
    Dense,
    Dropout,
    Activation,
    Flatten)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D)
from keras.optimizers import SGD

# Global variables
BATCH_SIZE = 100
IMG_SIZE = 28
NUM_LABELS = 10
VALIDATION_SIZE = 5000
NUM_CHANNELS = 1
NUM_EPOCHS = 1


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def main():
    savepath = './save_point'
    filepath = './save_point/keras_example_checkpoint.h5'

    # Extract MNIST dataset
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    train_data = extract_data(train_data_filename, 60000, dense=False)
    train_data = train_data.reshape((60000, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    train_labels = extract_labels(train_labels_filename, 60000, one_hot=True)
    test_data = extract_data(test_data_filename, 10000, dense=False)
    test_data = test_data.reshape((10000, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    test_labels = extract_labels(test_labels_filename, 10000, one_hot=True)

    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE, :]
    validation_set = (validation_data, validation_labels)
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:, ...]

    # Model construction
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
              input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Define optimizer and configure training process
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        nb_epoch=NUM_EPOCHS,
        batch_size=1000,
        validation_data=validation_set)

    print 'Save model weights'
    if not os.path.isdir (savepath):
        os.mkdir (savepath)
    model.save_weights(filepath, overwrite=True)

    predict = model.predict(test_data, batch_size=1000)

    print 'Test err: %.1f%%' % error_rate(predict, test_labels)

    print 'Test loss: %1.f%%, accuracy: %1.f%%', \
        tuple(model.evaluate(test_data, test_labels, batch_size=1000))

if __name__ == "__main__":
    main()

