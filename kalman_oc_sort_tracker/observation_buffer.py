from collections import deque

class ObservationBuffer:
    def __init__(self, max_length=10):
        self.buffer = deque(maxlen=max_length)

    def add(self, bbox):
        # [x, y, w, h]
        self.buffer.append(bbox)

    def get_linear_estimate(self, steps=1):
        if len(self.buffer) < 2:
            return self.buffer[-1] if self.buffer else None

        (x1, y1, w1, h1) = self.buffer[-2]
        (x2, y2, w2, h2) = self.buffer[-1]

        dx = x2 - x1
        dy = y2 - y1
        dw = w2 - w1
        dh = h2 - h1

        return [
            x2 + dx * steps,
            y2 + dy * steps,
            w2 + dw * steps,
            h2 + dh * steps
        ]