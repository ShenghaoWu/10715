import gzip
import os

import numpy as np

from six.moves import urllib

import torch as t

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 10000  # Size of the validation set.

# Download MNIST data
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.stat(filepath).st_size
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


# Extract the images
def extract_data(filename, num_images, norm_shift=False, norm_scale=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    #print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if norm_shift:
            data = data - (PIXEL_DEPTH / 2.0)
        if norm_scale:
            data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data

# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    #print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# Prepare MNISt data
def prepare_MNIST_data(n_train=10_000, n_val=10000, n_test=10000, 
                       use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    X_train = extract_data(train_data_filename, n_train+n_val, use_norm_shift, use_norm_scale)
    y_train = extract_labels(train_labels_filename, n_train+n_val)
    X_test = extract_data(test_data_filename, n_test, use_norm_shift, use_norm_scale)
    y_test = extract_labels(test_labels_filename, n_test)

    # Generate a validation set.
    X_val = X_train[:n_val, :]
    y_val = y_train[:n_val,:]

    X_train = X_train[n_val:n_val+n_train, :]
    y_train = y_train[n_val:n_val+n_train,:]

    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    assert len(X_train)==len(y_train)
    assert len(X_val)==len(y_val)
    assert len(X_test)==len(y_test)

    data = {'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test}
    return data

def get_dataloader(X, y, batch_size):
    X_tensor = t.tensor(X, dtype=t.float32)
    y_tensor = t.tensor(y, dtype=t.int64)
    dataset = t.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader