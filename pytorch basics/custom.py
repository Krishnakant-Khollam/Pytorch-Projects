import os
import random
import requests
import zipfile
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, Dict, List
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm.auto import tqdm
from torchinfo import summary

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Download and extract dataset
if not image_path.is_dir():
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak and sushi data...")
        zip_ref.extractall(image_path)
else:
    print(f"{image_path} directory already exists... skipping download.")


# Helper function to walk through directory
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


walk_through_dir(image_path)

# Define train and test directories
train_dir = image_path / "train"
test_dir = image_path / "test"

# Data transformations
data_transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

# Load datasets
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

# Dataloaders
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


# Define TinyVGG model
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


model = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)
summary(model, input_size=[1, 3, 64, 64])


# Training and testing functions
def train_step(model, dataloader, loss_fn, optimizer, device=device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += (y_pred.argmax(1) == y).sum().item() / len(y_pred)
    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_acc += (test_pred.argmax(1) == y).sum().item() / len(test_pred)
    return test_loss / len(dataloader), test_acc / len(dataloader)


# Full training loop
def train(
    model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device=device
):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(
            f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results


# Initialize model, loss function, and optimizer
torch.manual_seed(42)
model = TinyVGG(
    input_shape=3, hidden_units=10, output_shape=len(train_data.classes)
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
NUM_EPOCHS = 5
results = train(
    model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS, device
)


# Plot results
def plot_loss_curves(results):
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train loss")
    plt.plot(epochs, results["test_loss"], label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train accuracy")
    plt.plot(epochs, results["test_acc"], label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


plot_loss_curves(results)
