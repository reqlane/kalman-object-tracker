from kalman_filter import KalmanFilter
from observation_buffer import ObservationBuffer
from association import associate

class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.kf = KalmanFilter(bbox)
        self.buffer = ObservationBuffer()
        self.buffer.add(bbox)
        self.age = 1
        self.missed = 0
        self.active = True

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        self.kf.update(bbox)
        self.buffer.add(bbox)
        self.age += 1
        self.missed = 0
        self.active = True

    def mark_missed(self):
        self.missed += 1
        self.active = False

    def get_state(self):
        return self.kf.get_state()

class OCTracker:
    def __init__(self, iou_threshold=0.3, max_missed=30):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        predicted_boxes = [track.get_state() for track in self.tracks]
        matches, unmatched_detections, unmatched_tracks = associate(detections, predicted_boxes, self.iou_threshold)

        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        for idx in unmatched_detections:
            new_track = Track(self.next_id, detections[idx])
            self.tracks.append(new_track)

        for idx in unmatched_tracks:
            self.tracks[idx].mark_missed()

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        outputs = []
        min_age = 3

        for t in self.tracks:
            if t.active:
                if t.id is None and t.age >= min_age:
                    t.id = self.next_id
                    self.next_id += 1

                if t.id is not None:
                    x, y, w, h = t.get_state()
                    outputs.append((int(x), int(y), int(w), int(h), t.id))

        return outputs