from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="test1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="test1_model",
    patience=10,
)

print("학습 완료! runs/detect/test1_model 에 저장됨")
