import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================================================
# 1. Configuration
# =========================================================
BATCH_SIZE = 128
LATENT_DIM = 20
HIDDEN_DIM = 400
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "vae_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. Data loading
# =========================================================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 3. VAE model
# =========================================================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.relu(self.fc2(z))
        x_recon = self.sigmoid(self.fc3(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# =========================================================
# 4. Loss function
# =========================================================
def vae_loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(
        x_recon, x, reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss

# =========================================================
# 5. Initialize model and optimizer
# =========================================================
model = VAE(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================================
# 6. Training function
# =========================================================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        images = images.view(images.size(0), -1)  # flatten: [B, 1, 28, 28] -> [B, 784]

        optimizer.zero_grad()

        x_recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss_function(x_recon, images, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon / len(dataloader.dataset)
    avg_kl = total_kl / len(dataloader.dataset)

    return avg_loss, avg_recon, avg_kl

# =========================================================
# 7. Evaluation function
# =========================================================
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            images = images.view(images.size(0), -1)

            x_recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss_function(x_recon, images, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon / len(dataloader.dataset)
    avg_kl = total_kl / len(dataloader.dataset)

    return avg_loss, avg_recon, avg_kl

# =========================================================
# 8. Save reconstruction images
# =========================================================
def save_reconstructed_images(model, dataloader, device, epoch, output_dir):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        flat_images = images.view(images.size(0), -1)

        x_recon, _, _ = model(flat_images)

        original = images[:8]
        reconstructed = x_recon.view(-1, 1, 28, 28)[:8]

        comparison = torch.cat([original, reconstructed], dim=0)
        utils.save_image(
            comparison.cpu(),
            os.path.join(output_dir, f"reconstruction_epoch_{epoch}.png"),
            nrow=8
        )

# =========================================================
# 9. Save generated samples
# =========================================================
def save_generated_images(model, device, epoch, output_dir, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z).view(-1, 1, 28, 28)

        utils.save_image(
            samples.cpu(),
            os.path.join(output_dir, f"generated_epoch_{epoch}.png"),
            nrow=8
        )

# =========================================================
# 10. Training loop
# =========================================================
train_losses = []
test_losses = []
train_recon_losses = []
test_recon_losses = []
train_kl_losses = []
test_kl_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, DEVICE)
    test_loss, test_recon, test_kl = evaluate(model, test_loader, DEVICE)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_recon_losses.append(train_recon)
    test_recon_losses.append(test_recon)
    train_kl_losses.append(train_kl)
    test_kl_losses.append(test_kl)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}")
    print(f"  Test  Loss: {test_loss:.4f}  | Recon: {test_recon:.4f}  | KL: {test_kl:.4f}")

    save_reconstructed_images(model, test_loader, DEVICE, epoch, OUTPUT_DIR)
    save_generated_images(model, DEVICE, epoch, OUTPUT_DIR, LATENT_DIM)

# =========================================================
# 11. Save model
# =========================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mnist_vae.pth"))
print("Model saved successfully.")

# =========================================================
# 12. Plot loss curves
# =========================================================
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Total Loss")
plt.plot(epochs, test_losses, label="Test Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Total Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "total_loss_curve.png"))
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_recon_losses, label="Train Recon Loss")
plt.plot(epochs, test_recon_losses, label="Test Recon Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Reconstruction Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_loss_curve.png"))
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_kl_losses, label="Train KL Loss")
plt.plot(epochs, test_kl_losses, label="Test KL Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE KL Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "kl_loss_curve.png"))
plt.show()