import numpy as np

class KalmanFilter:
    def __init__(self, bbox):
        # [x, y, w, h]
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        s = bbox[2] * bbox[3]
        r = bbox[2] / bbox[3]

        # [cx, cy, s, r, vx, vy, vs]
        self.X = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)

        self.dt = 1.0
        self.F = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.F[i, i+4] = self.dt

        self.H = np.eye(4, 7, dtype=np.float32)  # [cx, cy, s, r]
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.Q = np.eye(7, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, bbox):
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        s = bbox[2] * bbox[3]
        r = bbox[2] / bbox[3]
        z = np.array([cx, cy, s, r], dtype=np.float32)

        y = z - np.dot(self.H, self.X)
        s = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        kg = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s)) # kalman gain

        self.X += np.dot(kg, y)
        self.P = np.dot(np.eye(7) - np.dot(kg, self.H), self.P)

    def adjust(self, linear_estimate):
        lx, ly, lw, lh = linear_estimate
        if not lh == 0:
            delta = np.array([lx, ly, lw * lh, lw / lh], dtype=np.float32) - np.dot(self.H, self.X)
            adjustment_gain = np.eye(4, dtype=np.float32) * 0.1
            self.X[:4] += np.dot(adjustment_gain, delta)
            self.P[:4, :4] += np.eye(4, dtype=np.float32) * 0.01

    def get_state(self):
        # [x, y, w, h]
        cx, cy, s, r = self.X[0], self.X[1], self.X[2], self.X[3]
        try:
            w = np.sqrt(max(1e-6, s * r))
            h = max(1e-6, s / w)
        except Exception:
            w, h = 1.0, 1.0
        x = cx - w / 2
        y = cy - h / 2

        return [x, y, w, h]