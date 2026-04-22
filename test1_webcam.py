import cv2
from ultralytics import YOLO

REAL_WIDTH_CM = 5.0
CALIB_DISTANCE_CM = 30.0

model = YOLO("runs/detect/test1_model/weights/best.pt")
cap = cv2.VideoCapture(0)

focal_length = None
calibrated = False

print("캘리브레이션: case를 카메라에서 30cm 거리에 들고 Space를 누르세요")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    h, w = frame.shape[:2]

    if not calibrated:
        cv2.putText(annotated, "case를 30cm 거리에 들고 Space 누르세요",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pixel_width = x2 - x1

        if calibrated and cls_name == "case" and pixel_width > 0:
            distance = (REAL_WIDTH_CM * focal_length) / pixel_width
            real_x = (cx - w / 2) * distance / focal_length
            real_y = (cy - h / 2) * distance / focal_length

            label = f"X:{real_x:.1f} Y:{real_y:.1f} Z:{distance:.1f}cm"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow("test1 webcam position", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and not calibrated:
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name == "case":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1
                if pixel_width > 0:
                    focal_length = (pixel_width * CALIB_DISTANCE_CM) / REAL_WIDTH_CM
                    calibrated = True
                    print(f"캘리브레이션 완료! 초점거리: {focal_length:.1f}px")
                    break
        if not calibrated:
            print("case가 감지되지 않았어요. 다시 시도하세요.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
