import torch
from torch import nn
from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

# Set device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Define helper functions for plotting decision boundaries
def plot_decision_boundary(model, X, y):
    """Plot decision boundary for binary classification."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(
        torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01)
    )
    grid_points = torch.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        probs = torch.sigmoid(model(grid_points.to(device)))
    zz = probs.view(xx.shape)
    plt.contourf(xx, yy, zz.cpu(), alpha=0.6, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")


# 1. Create the dataset
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Define the model (simplified for binary classification)
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        return self.model(x)


# Initialize the model, loss function, and optimizer
model = CircleModel().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy_fn = Accuracy().to(device)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_logits = model(X_train.to(device)).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train.to(device))
    acc = accuracy_fn(y_pred, y_train.to(device))

    # Backward pass
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test.to(device))
        test_acc = accuracy_fn(test_pred, y_test.to(device))

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

# Plot decision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()

# 4. Multi-Class Classification with Blobs
NUM_CLASSES = 4
X_blob, y_blob = make_blobs(
    n_samples=1000, n_features=2, centers=NUM_CLASSES, cluster_std=1.5, random_state=42
)
X_blob = torch.tensor(X_blob, dtype=torch.float32)
y_blob = torch.tensor(y_blob, dtype=torch.long)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=42
)


# Define model for multi-class classification
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.model(x)


# Initialize model, loss function, and optimizer for multi-class classification
model_blob = BlobModel(input_features=2, output_features=NUM_CLASSES).to(device)
loss_fn_blob = nn.CrossEntropyLoss()
optimizer_blob = torch.optim.SGD(model_blob.parameters(), lr=0.1)

# Training loop for multi-class classification
epochs = 100
for epoch in range(epochs):
    model_blob.train()
    optimizer_blob.zero_grad()

    # Forward pass
    y_logits = model_blob(X_blob_train.to(device))
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

    # Calculate loss and accuracy
    loss = loss_fn_blob(y_logits, y_blob_train.to(device))
    acc = accuracy_fn(y_pred, y_blob_train.to(device))

    # Backward pass
    loss.backward()
    optimizer_blob.step()

    # Evaluate on test set
    model_blob.eval()
    with torch.no_grad():
        test_logits = model_blob(X_blob_test.to(device))
        test_preds = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
        test_loss = loss_fn_blob(test_logits, y_blob_test.to(device))
        test_acc = accuracy_fn(test_preds, y_blob_test.to(device))

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

# Plot decision boundaries for multi-class classification
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_blob, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_blob, X_blob_test, y_blob_test)
plt.show()
