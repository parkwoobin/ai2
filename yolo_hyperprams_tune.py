from ultralytics import YOLO

def train_model():
    # Load the model from the last checkpoint
    model = YOLO("D:\\VS_Code\\AIService2\\runs\\detect\\train106\\weights\\last.pt")

    # Train the model with the correct data
    model.train(
        data="D:\\VS_Code\\AIService2\\data.yaml",
        epochs=200,  # Total number of epochs to reach
        batch=16,  # Batch size
        imgsz=640,
        lr0=0.0005,  # Adjust learning rate
        optimizer='AdamW',
        augment=True,  # Enable data augmentation
        save_period=1,  # Save checkpoint every epoch
        resume=True
    )

if __name__ == '__main__':
    train_model()