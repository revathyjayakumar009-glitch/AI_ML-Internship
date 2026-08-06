from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"D:\AI-ML Internship\Day48\helmet_dataset\data.yaml",
    epochs=1,
    imgsz=320,
    batch=1,
    workers=0,
    device="cpu",
    plots=False
)