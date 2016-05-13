import numpy as np
import os

from utils import (
	maybe_download,
	extract_data,
	extract_labels)

from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import (
	Dense,
	Activation,
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
	savepath = './save_point'
	filepath = './save_point/model_api_checkpoint.h5'
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

	img = Input(shape=(1, 28, 28))
	conv1 = Convolution2D(32, 3, 3, border_mode='same')(img)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2_1 = Convolution2D(64, 3, 3, border_mode='same')(pool1)
	conv2_2 = Convolution2D(64, 5, 5, border_mode='same')(pool1)
	conv2_1 = Activation('relu')(conv2_1)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
	pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
	dense1 = Flatten()(pool2_1)
	dense2 = Flatten()(pool2_2)
	dense = merge([dense1, dense2], mode='concat', concat_axis=1)
	dense = Dense(512)(dense)
	dense = Activation('relu')(dense)
	dense = Dense(256)(dense)
	dense = Activation('relu')(dense)
	dense = Dense(10)(dense)
	output = Activation('softmax')(dense)

	model = Model(input=[img], output=[output])

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

	model.compile(
					optimizer=sgd,
					loss=['categorical_crossentropy'],
					metrics=["accuracy"])

	model.fit(
					[train_data],
					[train_labels],
					nb_epoch=1,
					verbose=1,
					batch_size=1000,
					validation_data=validation_set)

	print 'Save model weights'
	if not os.path.isdir (savepath):
		os.mkdir (savepath)
	model.save_weights(filepath, overwrite=True)


	predictions = model.predict([test_data],
	                            batch_size=1000)

	print 'Test error: %.1f%%' % error_rate(predictions, test_labels)

	print 'Test loss: %.14f, Test accurracy %.4f' % \
	      tuple(model.evaluate([test_data], [test_labels], batch_size=1000))


if __name__ == '__main__':
	main()
