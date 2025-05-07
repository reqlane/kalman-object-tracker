import numpy as np
from kalman_filter import KalmanFilter
from utils import compute_iou

class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.kalman = KalmanFilter(cx, cy)
        self.bbox = bbox
        self.age = 1
        self.missed = 0

    def predict(self):
        cx, cy = self.kalman.predict()
        w, h = self.bbox[2], self.bbox[3]
        x, y = int(cx - w / 2), int(cy - h / 2)
        self.bbox = (x, y, w, h)

        return self.bbox

    def update(self, bbox):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.kalman.update([cx, cy])
        self.bbox = bbox
        self.missed = 0
        self.age += 1

class Tracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        matches, unmatched_detections, unmatched_tracks = self._associate_detections(detections)

        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_detections:
            self.tracks.append(Track(self.next_id, detections[det_idx]))
            self.next_id += 1

        survived_tracks = []
        for idx, track in enumerate(self.tracks):
            if idx in [trk_idx for _, trk_idx in matches]:
                survived_tracks.append(track)
                continue
            track.missed += 1
            if track.missed <= self.max_age:
                survived_tracks.append(track)
        self.tracks = survived_tracks

        return [(t.id, t.bbox) for t in self.tracks]

    def _associate_detections(self, detections):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                iou_matrix[d, t] = compute_iou(det, trk.bbox)

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # greedy alignment
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            det_idx, trk_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_indices.append((det_idx, trk_idx))
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, trk_idx] = -1
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(trk_idx)

        return matched_indices, unmatched_detections, unmatched_tracks
