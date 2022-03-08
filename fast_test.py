from deepface import DeepFace
import cv2, sys

cap = cv2.VideoCapture(1)
if cap.isOpened() == False:
    sys.exit('Camera is not open')

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

