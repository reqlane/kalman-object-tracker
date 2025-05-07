import cv2
import os
import time
from detector import MotionDetector
from tracker import Tracker
from utils.utils import preload_frames


def play_from_video_file(frames):
    window_name = "Tracking"
    cv2.namedWindow(window_name)

    detector = MotionDetector()
    tracker = Tracker()

    frame_count = 0
    start_time = time.time()

    for frame in frames:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for track_id, (x, y, w, h) in tracks:
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1) # fps lock | comment to unlock fps to maximum

    avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS (video file): {avg_fps:.2f}")
    print(f"Track IDs count during video: {tracker.next_id - 1}")
    cv2.destroyAllWindows()

def play_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    window_name = "Tracking"
    cv2.namedWindow(window_name)

    detector = MotionDetector()
    tracker = Tracker()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for track_id, (x, y, w, h) in tracks:
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS (video file): {avg_fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("----------")
    print("Kalman simple tracker")
    print("1 — Camera")
    print("2 — Video file")
    choice = input("Enter 1/2: ")

    if choice == "1":
        play_from_camera()
    elif choice == "2":
        video_path = os.path.join("static", "vtest.avi")
        if not os.path.exists(video_path):
            print(f"File '{video_path}' not found.")
            return
        frames = preload_frames(video_path)
        if frames:
            play_from_video_file(frames)
    else:
        print("Wrong input.")


if __name__ == "__main__":
    main()
