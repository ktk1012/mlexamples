import numpy as np
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
NUM_LABELS = 2
VALIDATION_SIZE = 5000
NUM_CHANNELS = 1
NUM_EPOCHS = 1

def main():
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

    #TODO: Split into validation and training sets
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE, :]
    validation_set = (validation_data, validation_labels)
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:, ...]

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
    model.add(Dense(2))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(
        train_data,
        train_labels,
        nb_epoch=NUM_EPOCHS,
        batch_size=100,
        validation_data=validation_set)

    score = model.evaluate(test_data, train_labels, batch_size=100)

    print 'Score: %d' % score

if __name__ == "__main__":
    main()

