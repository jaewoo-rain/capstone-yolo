import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/test1_model/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("test1", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
