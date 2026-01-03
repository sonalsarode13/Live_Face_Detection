import cv2
import threading
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
processing = False

reference_img = cv2.imread("models/test.jpg")

def check_face(frame):
    global face_match, processing
    try:
        result = DeepFace.verify(
            frame,
            reference_img,
            enforce_detection=False
        )
        face_match = result['verified']
    except Exception:
        face_match = False
    processing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if counter % 30 == 0 and not processing:
        processing = True
        threading.Thread(
            target=check_face,
            args=(frame.copy(),),
            daemon=True
        ).start()

    counter += 1

    if face_match:
        cv2.putText(
            frame, "Matched", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame, "Not Matched", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
