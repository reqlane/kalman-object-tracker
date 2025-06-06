import cv2
import os
import time
from detectors.frame_difference_detector import FrameDifferenceDetector
from detectors.yolo_detector import YOLODetector
from utils.utils import preload_frames


def play_from_video_file(frames, detector):
    window_name = "Tracking"
    cv2.namedWindow(window_name)

    frame_count = 0
    start_time = time.time()

    for frame in frames:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1) # fps lock | comment to unlock fps to maximum

    avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS (video file): {avg_fps:.2f}")
    cv2.destroyAllWindows()

def play_from_camera(detector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    window_name = "Tracking"
    cv2.namedWindow(window_name)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS (camera): {avg_fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("----------")
    print("Detection simple tracker")
    print("Choose detector")
    print("1 — Frame differencing detector")
    print("2 — YOLO detector")
    detector_choice = input("Enter 1/2: ")

    detector = None
    if detector_choice == "1":
        detector = FrameDifferenceDetector()
    elif detector_choice == "2":
        detector = YOLODetector(conf_threshold=0.4)
    else:
        print("Wrong input.")

    print("----------")
    print("Detection simple tracker")
    print("Choose video source")
    print("1 — Camera")
    print("2 — Video file")
    source_choice = input("Enter 1/2: ")

    if source_choice == "1":
        play_from_camera(detector)
    elif source_choice == "2":
        video_path = os.path.join("static", "vtest.avi")
        if not os.path.exists(video_path):
            print(f"File '{video_path}' not found.")
            return
        frames = preload_frames(video_path)
        if frames:
            play_from_video_file(frames, detector)
    else:
        print("Wrong input.")


if __name__ == "__main__":
    main()
