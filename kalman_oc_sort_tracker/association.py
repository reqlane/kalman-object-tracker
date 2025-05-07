import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.utils import compute_iou

def associate(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = compute_iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T

    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(trackers)))

    for d, t in matched_indices:
        if iou_matrix[d, t] >= iou_threshold:
            matches.append((d, t))
            unmatched_detections.remove(d)
            unmatched_tracks.remove(t)

    return matches, unmatched_detections, unmatched_tracks