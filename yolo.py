from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Load model and parse config file

result = model.train(
    name = "s_640_dropout0",  # Model name
    data="./data.yaml", 
    epochs=300,
    patience=10, #작으면 빨리 끝날수도 있음
    batch_size=16, #클수록 좋음
    imgsz=640,  # Train with 640x640 images
    optimizer = 'AdamW',  # Select optimizer
    lr0 = 0.001,  # Initial learning rate (no batch_size scaling)
    dropout = 0.0,  # 너무 크면 과소 적합될수있음?? 적당한 0.1~0.2

    
    )  # Train the model