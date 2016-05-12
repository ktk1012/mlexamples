"""
utils.py - utility for extract mnist datasets
"""
import gzip

from six.moves import urllib

import numpy
import os
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = './data'
IMAGE_SIZE = 28
PIXEL_DEPTH = 255  # Pixel scales (typically 255)

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath



def extract_data(filename, num_images, dense=False):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images, one_hot=True):
  """Extract the labels into a 1-hot matrix [image index, label index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
  # Convert to dense 1-hot representation.
  #return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)
  num_labels = labels.shape[0]
  labels_one_hot = numpy.zeros((num_labels, 10))
  index_offset = numpy.arange(num_labels) * 10
  # labels_equal_1 = (labels == 6).astype(numpy.int)
  # labels_equal_7 = (labels == 9).astype(numpy.int)
  # where_1_7 = labels_equal_1 + labels_equal_7
  if one_hot:
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    labels = labels_one_hot
  return labels
