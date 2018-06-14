import numpy as np


class KalmanFilter(object):
    def __init__(self, dim_x=1, dim_z=1):
        """
        Init with size of parameters to filter
        :param dim_x: size of location matrix
        :param dim_z: size of velocity matrix
        """
        # static variables
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.F = np.eye(dim_x, dim_x)  # projection matrix from state to state
        self.Q = np.eye(dim_x, dim_x)  # noise to variance matrix
        self.R = np.eye(dim_z, dim_z)  # detection noise mat
        self.H = np.eye(dim_x, dim_z)  # state to location matrix

        # dynamic variables
        self.x = np.ones((dim_x, 1))   # estimation model
        self.P = np.eye(dim_x, dim_x)  # estimation var matrix
        self.S = np.eye(dim_x, dim_x)  # detection var matrix
        self.y = None  # difference from target to estimation
        self.k = None  # Kalman's gain

    @property
    def xh(self):
        return np.matmul(self.x, self.H)

    def predict(self):
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, bbox):
        self.y = bbox - np.matmul(self.H, self.x)
        self.S = self.R + np.matmul(np.matmul(self.H, self.P), self.H.T)
        try:
            self.k = np.matmul(self.P, np.matmul(self.H.T, np.linalg.inv(self.S)))
        except TypeError:
            print(self.S)
            print(np.invert(self.S))
            raise ValueError("AAAA")
        self.x = self.x + np.matmul(self.k, self.y)
        self.P = np.matmul(
            np.eye(self.dim_x, self.dim_x) - np.matmul(np.matmul(self.k, self.H), self.P),
            (np.eye(self.dim_x, self.dim_x) - np.matmul(self.k, self.H)).T
        ) + np.matmul(self.k, np.matmul(self.R, self.k.T))
