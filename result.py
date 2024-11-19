from ultralytics import YOLO

def evaluate_model():
    # Load the tuned model
    model = YOLO("D:\\VS_Code\\AIService2\\runs\\detect\\train106\\weights\\best.pt")  # Load the fine-tuned model

    # Evaluate the model on the validation dataset
    data_path = "D:\\VS_Code\\AIService2\\data.yaml"
    print(f"Using data file: {data_path}")

    # Evaluate the model on the validation dataset
    results = model.val(data=data_path)

    # Print the evaluation results
    print(f"Precision: {results.box.map:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"F1 Score: {results.box.f1.mean():.4f}")

if __name__ == '__main__':
    evaluate_model()