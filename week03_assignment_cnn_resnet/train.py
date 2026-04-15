import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# =========================================================
# 1. Configuration
# =========================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
NUM_CLASSES = 10
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. Data preprocessing
# =========================================================
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# =========================================================
# 3. Load CIFAR-10 dataset
# =========================================================
train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes

print("Classes:", class_names)
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# =========================================================
# 4. Define CNN model
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2)                               # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 5. Define ResNet model
# =========================================================
def build_resnet18_for_cifar10(num_classes=10):
    model = models.resnet18(weights=None)

    # Adapt for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# =========================================================
# 6. Training and evaluation functions
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# =========================================================
# 7. Save plots
# =========================================================
def save_plots(model_name, history, output_dir):
    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history["train_losses"]) + 1), history["train_losses"], marker='o', label='Train Loss')
    plt.plot(range(1, len(history["test_losses"]) + 1), history["test_losses"], marker='s', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history["train_accuracies"]) + 1), history["train_accuracies"], marker='o', label='Train Accuracy')
    plt.plot(range(1, len(history["test_accuracies"]) + 1), history["test_accuracies"], marker='s', label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name.upper()} Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy_curve.png"), dpi=300)
    plt.close()


# =========================================================
# 8. Save history
# =========================================================
def save_history(model_name, history, output_dir):
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


# =========================================================
# 9. Train a single model
# =========================================================
def train_model(model, model_name, train_loader, test_loader, device, num_epochs, learning_rate, output_dir):
    print("=" * 60)
    print(f"Start training: {model_name.upper()}")
    print("=" * 60)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "test_losses": [],
        "test_accuracies": []
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["test_losses"].append(test_loss)
        history["test_accuracies"].append(test_acc)

        print(f"[{model_name.upper()}] Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f}, Test  Accuracy: {test_acc:.2f}%")
        print("-" * 60)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # save best weights
    model.load_state_dict(best_model_wts)
    model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    torch.save(model.state_dict(), model_path)

    save_plots(model_name, history, output_dir)
    save_history(model_name, history, output_dir)

    print(f"Finished training {model_name.upper()}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Saved model to: {model_path}")
    print()

    return history, best_acc


# =========================================================
# 10. Main: first CNN, then ResNet
# =========================================================
def main():
    # Train CNN first
    cnn_model = SimpleCNN(num_classes=NUM_CLASSES)
    cnn_history, cnn_best_acc = train_model(
        model=cnn_model,
        model_name="cnn",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR
    )

    # Train ResNet second
    resnet_model = build_resnet18_for_cifar10(num_classes=NUM_CLASSES)
    resnet_history, resnet_best_acc = train_model(
        model=resnet_model,
        model_name="resnet",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR
    )

    # Final summary
    summary = {
        "cnn_best_test_accuracy": cnn_best_acc,
        "resnet_best_test_accuracy": resnet_best_acc
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 60)
    print("Training completed for both models")
    print("=" * 60)
    print(f"CNN Best Test Accuracy:    {cnn_best_acc:.2f}%")
    print(f"ResNet Best Test Accuracy: {resnet_best_acc:.2f}%")


if __name__ == "__main__":
    main()