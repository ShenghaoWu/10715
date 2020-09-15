import time
import numpy as np


class SoftSVM(object):
    def __init__(self, C):
        """
        Soft Support Vector Machine Classifier
        The soft SVM algorithm classifies data points of dimension `d` 
        (this dimension includes the b).
        into {-1, +1} classes. It receives a regularization paramter `C` that
        controls the margin penalization.
        """
        self.C = C

    def predict(self, X:  np.ndarray) ->  np.ndarray:
        """
        Input
        ----------
        X: numpy array of shape (n, d)

        Return
        ------
        y_hat: numpy array of shape (n, )
        """

        # TODO: Write the prediction method
        assert y_hat.shape==(len(X),),\
            f'Check your y_hat dimensions they should be {(len(X),)} and are {y_hat.shape}'
        return y_hat

    def subgradient(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        subgrad: numpy array of shape (n, )
        """

        # TODO: Compute the subgradient
        assert subgrad.shape==(X.shape[1],),\
            f'Check your y_hat dimensions they should be {(X.shape[1],)} and are {subgrad.shape}'
        return subgrad

    def get_batch(self, X: np.ndarray, y: np.ndarray,
                  batch_size: int):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )
        batch_size: integer

        Return
        ------
        batch_x: numpy array of shape (batch_size, d)
        batch_y: numpy array of shape (batch_size, )
        """

        # TODO: write the get_batch method
        assert len(batch_x)==len(batch_y), f'Check your batch dimensions'
        return batch_x, batch_y

    def loss(self, X, y):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        svm_loss: float
        """

        # TODO: write the soft svm loss that incorporates regularization and hinge loss
        return svm_loss
    
    def accuracy(self, X, y):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        accuracy: float
        """

        # TODO: write the accuracy method to evaluate
        return accuracy

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, 
              n_iterations: int, batch_size: int,
              display_step: int,
              learning_rate: float,
              random_seed: 1) -> None:
        
        # Check inputs
        assert len(X_train)==len(y_train)
        assert len(X_test)==len(y_test)
        assert np.array_equal(np.sort(np.unique(y_train)), np.array([-1, 1]))
        
        # Initialize model
        np.random.seed(random_seed)
        self.d = X_train.shape[1]
        self.w = np.random.normal(size=(self.d,))
        t = 0
        self.trajectories = {'train_accuracy': [], 'test_accuracy': [],
                             'train_loss': [], 'test_loss': [],
                             'iteration_time': []}

        for step in range(n_iterations):
            #TODO: Obtain sample with mistake
            
            # TODO: Update perceptron and check mistakes

            if (step % display_step == 0):
                # TODO: Store trajectories
                train_accuracy = self.accuracy(X_train, y_train)
                test_accuracy = self.accuracy(X_test, y_test)

                train_loss = self.loss(X_train, y_train)
                test_loss = self.loss(X_test, y_test)
