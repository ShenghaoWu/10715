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
        y_hat = np.sign(np.matmul(X, self.w)+1e-10)
        assert y_hat.shape==(len(X),),\
            f'Check your y_hat dimensions they should be {(len(X),)} and are {y_hat.shape}'
        return y_hat

    def subgradient(self, X: np.ndarray, y: np.ndarray) -> None:
        # subgrad = np.zeros(self.w.shape)
        # for i in range(len(X)):
        #     x_i = X[i,:]
        #     y_i = y[i]
        #     if (1 - y_i*np.matmul(x_i,self.w))>0:
        #         subgrad += self.C*x_i*y_i
        # #subgrad = subgrad/len(X)
        # subgrad = self.w - subgrad

        subgrad = self.w + self.C * np.mean(np.where(np.tile(y * (X @ self.w) >=1,
        								(self.d, 1)).T, 0, -np.tile(y, (self.d, 1)).T * X), axis=0)

        assert subgrad.shape==(X.shape[1],), \
            f'Check your y_hat dimensions they should be {(X.shape[1],)} and are {subgrad.shape}'
        return subgrad

    def get_batch(self, X: np.ndarray, y: np.ndarray,
                  batch_size: int):
        # TODO: write the get_batch method
        ids = np.random.choice(range(len(X)), size=batch_size, replace=False)
        batch_x = X[ids, :]
        batch_y = y[ids]
        assert len(batch_x)==len(batch_y), f'Check your batch dimensions'
        return batch_x, batch_y

    def loss(self, X, y):
        l2_regularization = (0.5 * np.sum(np.square(self.w)))
        hinge_loss = self.C * np.sum(np.maximum(np.zeros(len(X)), (1-np.squeeze(y)*np.matmul(X, self.w))))
        svm_loss = l2_regularization + hinge_loss
        return svm_loss
    
    def accuracy(self, X, y):
        accuracy = (y == self.predict(X))
        accuracy = 100 * np.mean(accuracy)
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
                             'train_loss': [], 'test_loss': [], 'time':[]}
        start = time.time()
        for step in range(n_iterations):
            # Obtain sample with mistake
            batch_x, batch_y = self.get_batch(X_train, y_train, batch_size)
            
            # Update perceptron and check mistakes
            subgrad = self.subgradient(batch_x, batch_y)
            self.w = self.w - learning_rate * subgrad

            if (step % display_step == 0):
                self.trajectories['train_accuracy'].append(self.accuracy(X_train, y_train))
                self.trajectories['train_loss'].append(self.loss(X_train, y_train)/len(X_train))
                self.trajectories['test_accuracy'].append(self.accuracy(X_test, y_test))
                self.trajectories['test_loss'].append(self.loss(X_test, y_test)/len(X_test))
                self.trajectories['time'].append(time.time()-start)
