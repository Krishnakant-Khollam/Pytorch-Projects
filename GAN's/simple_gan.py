import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
latent_dim = 100  # Size of noise vector
image_size = 28 * 28  # MNIST images are 28x28
batch_size = 64
learning_rate = 0.0002
epochs = 50

# Data preparation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize images to [-1, 1]
    ]
)

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, image_size),
            nn.Tanh(),  # Output values between -1 and 1
        )

    def forward(self, z):
        return self.model(z)


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output probability of being real
        )

    def forward(self, x):
        return self.model(x)


# Initialize models
generator = Generator(latent_dim, image_size)
discriminator = Discriminator(image_size)

# Loss and optimizers
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)

        # Flatten real images
        real_images = real_images.view(batch_size, -1)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)  # Random noise
        fake_images = generator(z)  # Generate fake images

        real_output = discriminator(real_images)  # Discriminator on real images
        fake_output = discriminator(
            fake_images.detach()
        )  # Discriminator on fake images

        loss_d_real = criterion(real_output, real_labels)
        loss_d_fake = criterion(fake_output, fake_labels)
        loss_d = loss_d_real + loss_d_fake

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        fake_output = discriminator(fake_images)  # Discriminator on fake images
        loss_g = criterion(fake_output, real_labels)  # Fool discriminator

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    print(
        f"Epoch [{epoch+1}/{epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
    )

# Generate and visualize samples
import matplotlib.pyplot as plt

z = torch.randn(16, latent_dim)  # Generate 16 samples
fake_images = generator(z).view(-1, 1, 28, 28).detach()  # Reshape for visualization

# Plot
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[i, 0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()
