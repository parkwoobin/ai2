from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Load model and parse config file

result = model.tune(
    data = "./data.yaml",
    iterations=100,
    epochs=10,
    batch_size=16,
    imgsz=640,
    lr0=0.001,
    optimizer='AdamW',
)