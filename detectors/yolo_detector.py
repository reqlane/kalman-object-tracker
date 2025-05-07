from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)[0]

        bboxes = []
        for box in results.boxes:
            if box.conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            bboxes.append([x, y, w, h])

        return bboxes