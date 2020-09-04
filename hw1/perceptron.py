import numpy as np


class Perceptron(object):
    def __init__(self, d):
        """
        Perceptron Classifier
        The perceptron algorithm classifies data points of dimensionality `d`
        into {-1, +1} classes.
        """
        self.d = d
        self.w = np.zeros((d, 1)) # Don't change this
        self.b = np.zeros((1, 1)) # Don't change this

    def predict(self, x:  np.ndarray) ->  np.ndarray:
        #TODO: Complete the predict method
        return y_hat

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        #TODO: Complete the update method
        assert self.w.shape==(self.d, 1),\
            f'Check your weight dimensions they should be {(self.d, 1)} and are {self.w.shape}'

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, iterations: int) -> None:
                
        t = 0
        #TODO: Write the algorithm and store the trajectories
        self.trajectories = {'train': [], 'test': []}
        while (YOUR_CONDITION) & (t < iterations):
            pass
