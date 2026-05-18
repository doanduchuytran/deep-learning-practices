from ultralytics import YOLO


def main():
    # You can change this to yolo11s.pt for better accuracy.
    # Use yolo11n.pt for faster training and lower GPU memory usage.
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="configs/helmet.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project="runs/train",
        name="helmet_yolo11n",
        pretrained=True,
        patience=20,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        close_mosaic=10,
        save=True,
        plots=True,
    )

    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()