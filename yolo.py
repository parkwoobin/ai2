from ultralytics import YOLO
import os

if __name__ == '__main__':
    data_path = "D:\\VS_Code\\AIService2\\data.yaml"
    print(f"Using data file: {data_path}")
    
    # Check if data.yaml file exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
    
    # Load model from the specified checkpoint
    model_path = "D:\\VS_Code\\AIService2\\runs\\detect\\s_640_dropout025_12\\weights\\best.pt"
    model = YOLO(model_path)  # Load model from the checkpoint

    result = model.train(
        name="s_640_dropout025_",  # Model name
        data=data_path, 
        epochs=300,  # Total epochs to reach (e.g., if you've done 100 epochs, this should be 400)
        patience=10,
        batch=32,  # Adjusted batch size
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        dropout=0.25,
        resume=True  # This will attempt to resume from the specified checkpoint
    )
