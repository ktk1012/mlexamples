"""
Simple tensorflow example
Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
With one 5 x 5 convolutional layer, and two multiple convolutional layers
; 5 x 5 and 3 x 3  convolutional, respectively.
Merge above two convolved output in dense shape and perform fully connected activation.
fc1 - fc2 - fc3(output layer).
Each fully connected layer has 512, 256, 10 neurons respectively.
For more informations about tensorflow, see https://www.tensorflow.org
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy
import tensorflow as tf
from utils import (
    maybe_download,
    extract_data,
    extract_labels)

import matplotlib.pyplot as plt
from PIL import Image

IMAGE_SIZE = 28  # Image size
NUM_CHANNELS = 1  # Number of image channel (e.g RGB or gray scale)
NUM_LABELS = 2  # The number of target labels
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1000  # Size of each training batch
NUM_EPOCHS = 1  # The number of epochs to training
EVAL_BATCH_SIZE = 1000  # Size of evaluation batch size
EVAL_FREQUENCY = 10  # Number of steps between evaluations.

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean ('train', True, 'If true execute model training routine,'
                      'otherwise execute filtered image extraction routine')
flags.DEFINE_string ('model', None, 'If trained data already exists, load it')
flags.DEFINE_string ('input', None, 'Source of image, if None use random images from MNIST dataset')

#TODO: Make conv image extraction flags.


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
      predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')


  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 60000, dense=False)
  train_labels = extract_labels(train_labels_filename, 60000, one_hot=True)
  test_data = extract_data(test_data_filename, 10000, dense=False )
  test_labels = extract_labels(test_labels_filename, 10000, one_hot=True)


  # Generate a validation set.
  validation_data = train_data[:VALIDATION_SIZE, ...]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, ...]
  train_labels = train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, NUM_LABELS))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}

  # First convolutional layer
  conv1_weights = tf.Variable(
      tf.truncated_normal([3, 3, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([32]))

  # Two second convolutional layers 5 x 5 filter, and 3 x 3 filters.
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.01, shape=[64]))

  conv2_weights2 = tf.Variable(
      tf.truncated_normal([3, 3, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases2 = tf.Variable(tf.constant(0.01, shape=[64]))

  # First fully connected layer after conv layer
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, 512],
          stddev=0.05,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.01, shape=[512]))

  # Second fully connected layer
  fc2_weights = tf.Variable(
      tf.truncated_normal([512, 256],
                          stddev=0.05,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[256]))

  # Output layer
  fc3_weights = tf.Variable(
      tf.truncated_normal([256, NUM_LABELS],
                          stddev=0.04,
                          seed=SEED))
  fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    if train:
        relu = tf.nn.dropout(relu, .5)
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    conv2 = tf.nn.conv2d(pool,
                         conv2_weights2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases2))

    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    pool2 = tf.nn.max_pool(relu2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool = tf.concat(3, [pool, pool2])
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc3_weights) + fc3_biases

  def extract_filter (data):
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu1,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    conv2 = tf.nn.conv2d(pool,
                         conv2_weights2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases2))

    return relu1, relu2, relu3


  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                  tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  saver = tf.train.Saver()
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    if FLAGS.model:
      saver.restore(sess, FLAGS.model)  # If model exists, load it
    else:
      sess.run(tf.initialize_all_variables())  # If there is no model randomly initialize
    if FLAGS.train:
      # Loop through training steps.
      for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph is should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = sess.run(
            [optimizer, loss, learning_rate, train_prediction],
            feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
          print('Validation error: %.1f%%' % error_rate(
              eval_in_batches(validation_data, sess), validation_labels))
          sys.stdout.flush()
      # Finally print the result!
      test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
      print('Test error: %.1f%%' % test_error)
      print ('Optimization done')
      print ('Save models')
      if not tf.gfile.Exists("./conv_save"):
          tf.gfile.MakeDirs("./conv_save")
      saver_path = saver.save(sess, "./conv_save/model.ckpt")
      print ('Successfully saved file: %s' % saver_path)
    else:  # If train flag is false, execute image extraction routine
      print ("Filter extraction routine")
      aa = train_data[1:2, :, :, :]
      print (aa.shape)
      # Run extract filter operations (conv1, conv2 and conv3 layers)
      images = sess.run(extract_filter(train_data[1:2, :, :, :]))
      print (images[2].shape)
      plt.imshow (images[2][0, :, :, 32] * 255 + 255 / 2, cmap='gray')
      # plt.imshow (images[2][0, :, :, 32], cmap='gray')
      plt.show ()
      # Save all outputs
      for i in range (3):
        filter_shape = images[i].shape
        img_size = [filter_shape[1], filter_shape[2]]
        print (img_size)
        # new_im = Image.new()


if __name__ == '__main__':
  tf.app.run()
