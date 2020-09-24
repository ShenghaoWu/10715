import numpy as np


class Perceptron(object):
    def __init__(self, d):
        """
        Perceptron Classifier
        The perceptron algorithm classifies data points of dimensionality `d`
        into {-1, +1} classes.
        """
        self.d = d
        self.w = np.zeros((d, 1))
        self.b = np.zeros((1, 1))

    def predict(self, x:  np.ndarray) ->  np.ndarray:
        y_hat = np.sign(np.dot(x, self.w) + self.b + 1e-16)
        return np.squeeze(y_hat)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.w = self.w + y * x.T
        self.b = self.b + y
        assert self.w.shape==(self.d, 1),\
            f'Check your weight dimensions they should be {(self.d, 1)} and are {self.w.shape}'

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, iterations: int) -> None:
                
        t = 0
        y_hat = self.predict(X_train)
        mistake_idxs = np.where((y_train * y_hat < 0))[0]
        
        # Train perceptron
        self.trajectories = {'train': [], 'test': []}
        while (len(mistake_idxs)>0) & (t < iterations):
            # Obtain sample with mistake
            idx = np.random.choice(mistake_idxs)
            x_t = X_train[[idx], :]
            y_t = y_train[idx]
            
            # Update perceptron and check mistakes
            self.update(x_t, y_t)
            y_hat = self.predict(X_train)
            y_hat_test = self.predict(X_test)

            train_errors = (y_train * y_hat < 0)
            test_errors = (y_test * y_hat_test < 0)
            self.trajectories['train'].append(100-100*np.mean(train_errors))
            self.trajectories['test'].append(100-100*np.mean(test_errors))

            mistake_idxs = np.where(train_errors)[0]
            t+=1