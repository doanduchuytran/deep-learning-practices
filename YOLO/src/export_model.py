from ultralytics import YOLO


def main():
    model_path = "runs/train/helmet_yolo11n/weights/best.pt"
    model = YOLO(model_path)

    # ONNX is useful for deployment and inference optimization.
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        simplify=True,
    )

    print("Model export finished.")


if __name__ == "__main__":
    main()