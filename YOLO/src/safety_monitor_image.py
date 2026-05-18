from pathlib import Path
import cv2
from ultralytics import YOLO


def draw_text(img, text, x, y, color=(0, 0, 255)):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    model_path = "runs/train/helmet_yolo11n/weights/best.pt"
    image_path = "dataset/test/images/example.jpg"

    output_dir = Path("runs/safety_monitor")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    names = model.names

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    unsafe_detected = False

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Customize these class names according to your dataset.
        if class_name.lower() in ["no_hardhat", "no-helmet", "no_helmet", "head"]:
            unsafe_detected = True
            color = (0, 0, 255)
            label = f"UNSAFE: {class_name} {conf:.2f}"
        elif class_name.lower() in ["helmet", "hardhat", "hard_hat"]:
            color = (0, 255, 0)
            label = f"SAFE: {class_name} {conf:.2f}"
        else:
            color = (255, 255, 0)
            label = f"{class_name} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_text(image, label, x1, max(y1 - 10, 20), color)

    if unsafe_detected:
        draw_text(image, "WARNING: Worker without helmet detected!", 30, 50, (0, 0, 255))
    else:
        draw_text(image, "No helmet violation detected", 30, 50, (0, 255, 0))

    output_path = output_dir / "safety_result.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()