from ultralytics import YOLO


def main():
    model_path = "runs/detect/runs/train/helmet_yolo11n/weights/best.pt"
    model = YOLO(model_path)

    metrics = model.val(
        data="configs/helmet.yaml",
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        project="runs/val",
        name="helmet_yolo11n_test",
        plots=True,
    )

    print("\nValidation results:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()