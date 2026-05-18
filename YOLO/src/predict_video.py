from ultralytics import YOLO


def main():
    model_path = "runs/train/helmet_yolo11n/weights/best.pt"

    # Change this to your own video file.
    source = "test_video.mp4"

    model = YOLO(model_path)

    model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        save=True,
        project="runs/predict",
        name="helmet_video",
    )

    print("Video prediction finished.")


if __name__ == "__main__":
    main()