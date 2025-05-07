def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_width = max(0, x_b - x_a)
    inter_height = max(0, y_b - y_a)
    inter_area = inter_width * inter_height

    box_a_area = box_a[2] * box_a[3]
    box_b_area = box_b[2] * box_b[3]

    union_area = box_a_area + box_b_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area
