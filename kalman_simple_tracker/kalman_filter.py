import numpy as np

class KalmanFilter:
    def __init__(self, x, y):
        # [x, y, vx, vy]
        self.state = np.array([[x], [y], [0], [0]], dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 1000.0

        self.F = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],   # x
            [0, 1, 0, 0]    # y
        ], dtype=np.float32)

        self.Q = np.eye(4, dtype=np.float32) * 1.0

        self.R = np.eye(2, dtype=np.float32) * 10.0

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.state[:2].flatten()  # [x, y]

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape((2, 1))
        y = z - np.dot(self.H, self.state)
        s = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        kg = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s))  # kalman gain

        self.state = self.state + np.dot(kg, y)
        i = np.eye(self.F.shape[0], dtype=np.float32)
        self.P = (i - np.dot(kg, self.H)) @ self.P
