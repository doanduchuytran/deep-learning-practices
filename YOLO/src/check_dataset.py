ifrom pathlib import Path
import yaml


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def check_split(dataset_root: Path, split_name: str, image_dir: str, num_classes: int):
    image_path = dataset_root / image_dir
    label_path = dataset_root / image_dir.replace("images", "labels")

    print(f"\nChecking split: {split_name}")
    print(f"Image folder: {image_path}")
    print(f"Label folder: {label_path}")

    if not image_path.exists():
        print(f"[ERROR] Image folder does not exist: {image_path}")
        return

    if not label_path.exists():
        print(f"[ERROR] Label folder does not exist: {label_path}")
        return

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in image_extensions:
        images.extend(list(image_path.glob(f"*{ext}")))

    print(f"Number of images: {len(images)}")

    missing_labels = 0
    empty_labels = 0
    invalid_lines = 0

    for img in images:
        label_file = label_path / f"{img.stem}.txt"

        if not label_file.exists():
            missing_labels += 1
            continue

        lines = label_file.read_text().strip().splitlines()

        if len(lines) == 0:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_lines += 1
                print(f"[INVALID FORMAT] {label_file}: {line}")
                continue

            try:
                class_id = int(float(parts[0]))
                values = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_lines += 1
                print(f"[INVALID VALUE] {label_file}: {line}")
                continue

            if class_id < 0 or class_id >= num_classes:
                invalid_lines += 1
                print(f"[INVALID CLASS ID] {label_file}: {line}")

            for v in values:
                if v < 0.0 or v > 1.0:
                    invalid_lines += 1
                    print(f"[INVALID BOX VALUE] {label_file}: {line}")
                    break

    print(f"Missing labels: {missing_labels}")
    print(f"Empty labels: {empty_labels}")
    print(f"Invalid lines: {invalid_lines}")

    if missing_labels == 0 and invalid_lines == 0:
        print("[OK] Dataset split looks valid.")


def main():
    yaml_path = "configs/helmet.yaml"
    data = load_yaml(yaml_path)

    dataset_root = Path("configs") / data["path"]
    dataset_root = dataset_root.resolve()

    names = data["names"]
    num_classes = len(names)

    print("Dataset root:", dataset_root)
    print("Classes:", names)
    print("Number of classes:", num_classes)

    check_split(dataset_root, "train", data["train"], num_classes)
    check_split(dataset_root, "val", data["val"], num_classes)

    if "test" in data:
        check_split(dataset_root, "test", data["test"], num_classes)


if __name__ == "__main__":
    main()