import numpy as np

from utils import (
    maybe_download,
    extract_data,
    extract_labels)

from keras.models import Graph
from keras.layers.core import (
    Dense,
    Dropout,
    Activation,
    Merge,
    Flatten)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D)
from keras.optimizers import SGD


BATCH_SIZE = 100
IMG_SIZE = 28
NUM_LABELS = 2
VALIDATION_SIZE = 5000
NUM_CHANNELS = 1
NUM_EPOCHS = 5


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def main():
    filepath = './save_point/checkpoint'
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

    graph = Graph()
    graph.add_input(name='input1', input_shape=(NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    graph.add_node(
        Convolution2D(32, 3, 3, border_mode='same'),
        name='conv1', input='input1')
    graph.add_node(
        Activation('relu'),
        name='conv1-relu', input='conv1')
    graph.add_node(
        MaxPooling2D(pool_size=(2, 2)),
        name='pool1', input='conv1-relu')
    graph.add_node(
        Convolution2D(64, 3, 3, border_mode='same'),
        name='conv2-1', input='pool1')
    graph.add_node(
        Activation('relu'),
        name='conv2-1-relu', input='conv2-1')
    graph.add_node(
        Convolution2D(64, 5, 5, border_mode='same'),
        name='conv2-2', input='pool1')
    graph.add_node(
        Activation('relu'),
        name='conv2-2-relu', input='conv2-2')
    graph.add_node(
        MaxPooling2D(pool_size=(2, 2)),
        name='pool2-1', input='conv2-1-relu')
    graph.add_node(
        MaxPooling2D(pool_size=(2, 2)),
        name='pool2-2', input='conv2-2-relu')
    graph.add_node(
        Flatten(),
        name='dense',
        inputs=['pool2-1', 'pool2-2'],
        )
    graph.add_node(
        Dense(512),
        name='fc1',
        input='dense')
    graph.add_node(
        Activation('relu'),
        name='fc1-relu',
        input='fc1')
    graph.add_node(
        Dense(256),
        name='fc2',
        input='fc1-relu')
    graph.add_node(
        Activation('relu'),
        name='fc2-relu',
        input='fc2')
    graph.add_node(
        Dense(10),
        name='fc3',
        input='fc2-relu')
    graph.add_node(
        Activation('softmax'),
        name='out',
        input='fc3')
    graph.add_output(
        name='output',
        input='out')

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

    graph.compile(
        optimizer=sgd, loss={'output': 'categorical_crossentropy'}, metrics=["accuracy"])


    graph.fit({'input1': train_data, 'output': train_labels}, nb_epoch=1, verbose=1, batch_size=1000)
    predictions = graph.predict({'input1': test_data, 'output': test_labels}, batch_size=100)
    print 'Test error: %.1f%%' % error_rate(predictions, test_labels)

if __name__ == '__main__':
    main()


