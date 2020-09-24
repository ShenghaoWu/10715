import os
import gzip
import urllib
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "./data/"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 10000  # Size of the validation set.

def maybe_download(filename):
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.stat(filepath).st_size
        print('Successfully downloaded', filename, size, 'bytes.')

# Extract the images
def extract_data(filename):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    if filename=='train-images-idx3-ubyte.gz':
        num_images = 60_000
    elif filename=='t10k-images-idx3-ubyte.gz':
        num_images = 10_000
    
    filepath = os.path.join(DATA_DIRECTORY, filename)
    with gzip.open(filepath) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

# Extract the labels
def extract_labels(filename):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    if filename=='train-labels-idx1-ubyte.gz':
        num_images = 60_000
    elif filename=='t10k-labels-idx1-ubyte.gz':
        num_images = 10_000
    
    filepath = os.path.join(DATA_DIRECTORY, filename)
    with gzip.open(filepath) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
