import torch
import torchvision
import matplotlib.pyplot as plt
import random
import pandas as pd
from torch import nn
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# DataLoader setup
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Display sample image
image, label = train_data[0]
class_names = train_data.classes
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)


# Model definition
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# Model initialization
model_2 = FashionMNISTModelV2(
    input_shape=1, hidden_units=10, output_shape=len(class_names)
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)


# Train and evaluation loop functions
def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    return train_loss / len(data_loader), train_acc / len(data_loader)


def test_step(model, data_loader, loss_fn, accuracy_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    return test_loss / len(data_loader), test_acc / len(data_loader)


# Training loop
epochs = 3
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    print(f"Epoch: {epoch + 1}")
    train_loss, train_acc = train_step(
        model_2, train_dataloader, loss_fn, optimizer, accuracy_fn, device
    )
    test_loss, test_acc = test_step(
        model_2, test_dataloader, loss_fn, accuracy_fn, device
    )
    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )


# Evaluate model
def eval_model(model, data_loader, loss_fn, accuracy_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            total_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    return total_loss / len(data_loader), total_acc / len(data_loader)


model_2_results = eval_model(model_2, test_dataloader, loss_fn, accuracy_fn, device)
print(
    f"Model Test Loss: {model_2_results[0]:.4f} | Test Acc: {model_2_results[1]:.2f}%"
)

# Save model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "fashion_mnist_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(model_2.state_dict(), MODEL_SAVE_PATH)

# Load model and evaluate
loaded_model_2 = FashionMNISTModelV2(
    input_shape=1, hidden_units=10, output_shape=len(class_names)
).to(device)
loaded_model_2.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_2_results = eval_model(
    loaded_model_2, test_dataloader, loss_fn, accuracy_fn, device
)
print(
    f"Loaded Model Test Loss: {loaded_model_2_results[0]:.4f} | Test Acc: {loaded_model_2_results[1]:.2f}%"
)

# Confusion matrix and plot
y_preds = []
with torch.no_grad():
    for X, _ in test_dataloader:
        X = X.to(device)
        y_pred = model_2(X)
        y_preds.append(y_pred.argmax(dim=1))
y_pred_tensor = torch.cat(y_preds)

confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass").to(device)
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets.to(device))

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.cpu().numpy(), class_names=class_names, figsize=(10, 7)
)
