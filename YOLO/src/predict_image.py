from ultralytics import YOLO


def main():
    model_path = "runs/train/helmet_yolo11n/weights/best.pt"
    source = "dataset/test/images"

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        save=True,
        project="runs/predict",
        name="helmet_images",
    )

    print("Prediction finished.")
    print(f"Number of results: {len(results)}")


if __name__ == "__main__":
    main()