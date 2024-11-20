import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# Check PyTorch version
print(torch.__version__)

# Create known parameters
weight = 0.7
bias = 0.3
start, end, step = 0, 1, 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})


# Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))

    def forward(self, x):
        return self.weights * x + self.bias


# Check model parameters
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))

# Predictions
with torch.inference_mode():
    y_preds = model_0(X_test)
plot_predictions(predictions=y_preds)

# Train
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 200
loss_values = []  # Initialize loss_values
test_loss_values = []  # Initialize test_loss_values
train_accuracy_values = []  # Initialize training accuracy values
test_accuracy_values = []  # Initialize testing accuracy values


# Accuracy function
def calculate_accuracy(y_true, y_pred):
    # Define a threshold for accuracy
    threshold = 0.1  # Acceptable error margin
    correct_predictions = torch.abs(y_true - y_pred) < threshold
    return correct_predictions.float().mean().item()


for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

        # Calculate accuracies
        train_accuracy = calculate_accuracy(y_train, y_pred)
        test_accuracy = calculate_accuracy(y_test, test_pred)

    if epoch % 10 == 0:
        loss_values.append(loss.item())  # Store loss value
        test_loss_values.append(test_loss.item())  # Store test loss value
        train_accuracy_values.append(train_accuracy)  # Store train accuracy
        test_accuracy_values.append(test_accuracy)  # Store test accuracy
        print(
            f"Epoch: {epoch} | Loss: {loss.item()} | Test loss: {test_loss.item()} | Train accuracy: {train_accuracy:.4f} | Test accuracy: {test_accuracy:.4f}"
        )

# Plot loss
epoch_count = list(range(0, epochs, 10))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.legend()
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epoch_count, train_accuracy_values, label="Train accuracy")
plt.plot(epoch_count, test_accuracy_values, label="Test accuracy")
plt.legend()
plt.title("Accuracy over epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Save model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# Load model
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# Compare predictions
comparison = (y_preds == loaded_model_preds).all()
print(
    f"Are original predictions equal to loaded model predictions? {comparison.item()}"
)

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model_0.to(device)
X_train, y_train, X_test, y_test = (
    X_train.to(device),
    y_train.to(device),
    X_test.to(device),
    y_test.to(device),
)
