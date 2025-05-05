import cv2
import os
from detector import MotionDetector

def main():
    print("1 — Camera")
    print("2 — Video file")
    choice = input("Enter 1/2: ")

    if choice == "1":
        cap = cv2.VideoCapture(0)
    elif choice == "2":
        video_path = os.path.join("static", "vtest.avi")
        if not os.path.exists(video_path):
            print(f"File '{video_path}' not found.")
            return
        cap = cv2.VideoCapture(video_path)
    else:
        print("Wrong input.")
        return

    if not cap.isOpened():
        print("Couldn't open video.")
        return

    detector = MotionDetector()

    window_name = "Video"
    cv2.namedWindow(window_name)
    cv2.waitKey(1)

    print("Video started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        detections = detector.detect(frame)

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()