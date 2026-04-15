import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# 2. Configuration
# =========================================================
CSV_PATH = "data/imdb_reviews.csv"   # change to your csv file name

VOCAB_SIZE = 20000
MAX_LEN = 200
BATCH_SIZE = 64

EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# =========================================================
# 3. Text cleaning and tokenization
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)         # remove HTML line breaks
    text = re.sub(r"[^a-z0-9\s']", " ", text)      # keep letters, digits, spaces, apostrophe
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text


def tokenize(text):
    return text.split()


# =========================================================
# 4. Convert sentiment labels to numeric
# =========================================================
def convert_label(x):
    x = str(x).lower().strip()

    if x in ["positive", "pos", "1"]:
        return 1
    elif x in ["negative", "neg", "0"]:
        return 0
    else:
        return None


# =========================================================
# 5. Build vocabulary from training set only
# =========================================================
def build_vocab(texts, vocab_size):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    most_common = counter.most_common(vocab_size - 2)

    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


# =========================================================
# 6. Convert text to token IDs
# =========================================================
def encode_text(text, word2idx, max_len):
    tokens = tokenize(text)
    token_ids = [word2idx.get(token, word2idx[UNK_TOKEN]) for token in tokens]

    # truncate
    token_ids = token_ids[:max_len]

    # pad
    if len(token_ids) < max_len:
        token_ids += [word2idx[PAD_TOKEN]] * (max_len - len(token_ids))

    return token_ids


# =========================================================
# 7. Custom Dataset
# =========================================================
class ReviewDataset(Dataset):
    def __init__(self, dataframe, word2idx, max_len):
        self.texts = dataframe["clean_review"].tolist()
        self.labels = dataframe["label"].tolist()
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = encode_text(text, self.word2idx, self.max_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


# =========================================================
# 8. LSTM sentiment classifier
# =========================================================
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, max_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, max_len, embed_dim)

        output, (h_n, c_n) = self.lstm(embedded)

        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]
        # last_hidden shape: (batch_size, hidden_dim)

        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(1)
        # logits shape: (batch_size,)

        return logits


# =========================================================
# 9. Metrics
# =========================================================
def binary_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).float().sum()
    acc = correct / len(labels)
    return acc.item()


# =========================================================
# 10. Train / evaluate functions
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += binary_accuracy(logits.detach(), labels)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += binary_accuracy(logits, labels)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc


# =========================================================
# 11. Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    print("First 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("Original shape:", df.shape)

    # -----------------------------------------------------
    # Keep only needed columns
    # -----------------------------------------------------
    df = df[["review", "sentiment"]]

    # -----------------------------------------------------
    # Remove missing values
    # -----------------------------------------------------
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

    # -----------------------------------------------------
    # Convert labels
    # -----------------------------------------------------
    df["label"] = df["sentiment"].apply(convert_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # -----------------------------------------------------
    # Clean text
    # -----------------------------------------------------
    df["clean_review"] = df["review"].apply(clean_text)

    print("\nShape after cleaning:", df.shape)
    print(df[["review", "sentiment", "label", "clean_review"]].head())

    # -----------------------------------------------------
    # Split train / val / test
    # -----------------------------------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    print(f"\nTrain size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # -----------------------------------------------------
    # Build vocabulary from train only
    # -----------------------------------------------------
    word2idx, idx2word = build_vocab(train_df["clean_review"].tolist(), VOCAB_SIZE)

    print(f"Vocabulary size: {len(word2idx)}")
    print("First 20 vocab items:")
    print(list(word2idx.items())[:20])

    # -----------------------------------------------------
    # Create datasets
    # -----------------------------------------------------
    train_dataset = ReviewDataset(train_df, word2idx, MAX_LEN)
    val_dataset = ReviewDataset(val_df, word2idx, MAX_LEN)
    test_dataset = ReviewDataset(test_df, word2idx, MAX_LEN)

    # -----------------------------------------------------
    # Create dataloaders
    # -----------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------
    # Inspect one batch
    # -----------------------------------------------------
    for input_ids, labels in train_loader:
        print("\nOne batch check:")
        print("input_ids shape:", input_ids.shape)
        print("labels shape:", labels.shape)
        print("first review token ids:", input_ids[0][:20])
        print("first label:", labels[0].item())
        break

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------
    model = LSTMSentimentClassifier(
        vocab_size=len(word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=word2idx[PAD_TOKEN]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_imdb_lstm_model.pth")

    print("\nBest validation accuracy:", best_val_acc)
    print("Best model saved to best_imdb_lstm_model.pth")

    # -----------------------------------------------------
    # Load best model and test
    # -----------------------------------------------------
    model.load_state_dict(torch.load("best_imdb_lstm_model.pth", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # -----------------------------------------------------
    # Plot loss
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, marker='s', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM IMDb CSV - Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("imdb_lstm_csv_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------
    # Plot accuracy
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, marker='s', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM IMDb CSV - Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("imdb_lstm_csv_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Plots saved: imdb_lstm_csv_loss.png, imdb_lstm_csv_accuracy.png")


if __name__ == "__main__":
    main()