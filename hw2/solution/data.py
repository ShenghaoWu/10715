import numpy as np
from sklearn.datasets import make_blobs


def get_data():
    std = 3.1
    train_data = make_blobs(n_samples=10_000, n_features=2, centers=2, cluster_std=std, random_state=1)
    test_data = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=std, random_state=1)
    return train_data, test_data

def prepare_data(data):
    X = np.hstack([np.ones((len(data[0]),1)), data[0]])
    y = 1*(data[1]==1) -1*(data[1]==0)
    return X, y
