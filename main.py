import cv2
import numpy as np

def main():
    print("✅ OpenCV version:", cv2.__version__)
    print("✅ NumPy version:", np.__version__)

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(img, "OpenCV Works!", (75, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Test Window", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()