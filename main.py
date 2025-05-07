import cv2
import os
from tracker import Tracker
from detector import MotionDetector

def main():
    print("1 — Camera")
    print("2 — Video file")
    choice = input("Enter 1/2: ")

    if choice == "1":
        cap = cv2.VideoCapture(0)
    elif choice == "2":
        path = os.path.join("static", "vtest.avi")
        if not os.path.exists(path):
            print(f"File {path} not found.")
            return
        cap = cv2.VideoCapture(path)
    else:
        print("Wrong input")
        return

    if not cap.isOpened():
        print("Couldn't open video.")
        return

    detector = MotionDetector()
    tracker = Tracker()

    window_name = "Tracking"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for track_id, (x, y, w, h) in tracks:
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)

        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()