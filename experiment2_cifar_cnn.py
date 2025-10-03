# Set up SSL context for dataset download (for environments with SSL issues)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Use albumentations for advanced augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Compute mean and std for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Albumentations transform for training
# Use only the arguments supported by your version of albumentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    # Use only supported arguments for your albumentations version
    A.CoarseDropout(p=0.5),
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ToTensorV2(),
])

# Albumentations transform for test (only normalization)
test_transform = A.Compose([
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ToTensorV2(),
])

# Custom dataset wrapper to use albumentations with torchvision datasets
from torch.utils.data import Dataset
from PIL import Image

class AlbumentationsCIFAR10(Dataset):
    def __init__(self, torchvision_dataset, transform=None):
        self.dataset = torchvision_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# Replace torchvision transforms with albumentations applies horizontal flip, shiftScaleRotate, coarseDropout
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

trainset = AlbumentationsCIFAR10(trainset, transform=train_transform)
testset = AlbumentationsCIFAR10(testset, transform=test_transform)

# Hyperparameters
batch_size = 64

# Data loaders for batching and shuffling
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Block 1: Initial feature extraction
        # Input: 3x32x32, Output: 16x32x32, RF: 3x3
        # Effect: Extracts low-level features (edges, colors) from input image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2: Depthwise separable conv for efficient feature extraction
        # Input: 16x32x32, Output: 32x32x32, RF: 3x3 (depthwise) + 1x1 (pointwise)
        # Effect: Efficiently increases feature complexity while keeping parameter count low
        self.sep_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),  # depthwise
            nn.Conv2d(16, 32, kernel_size=1)  # pointwise
        )
        self.bn_sep1 = nn.BatchNorm2d(32)

        # Block 3: Dilated conv to increase receptive field
        # Input: 32x32x32, Output: 32x32x32, RF: 5x5 (dilated)
        # Effect: Captures larger spatial context without increasing parameters
        self.dil_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.bn_dil1 = nn.BatchNorm2d(32)

        # Block 4: Downsampling with stride, keeps channels
        # Input: 32x32x32, Output: 32x16x16, RF: 3x3, stride=2
        # Effect: Reduces spatial resolution (downsamples) to focus on higher-level features
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 5: Depthwise separable conv at lower resolution
        # Input: 32x16x16, Output: 32x16x16, RF: 3x3 (depthwise) + 1x1 (pointwise)
        # Effect: Further increases feature complexity efficiently at lower resolution
        self.sep_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.bn_sep2 = nn.BatchNorm2d(32)

        # Block 6: Dilated conv at lower resolution
        # Input: 32x16x16, Output: 32x16x16, RF: 5x5 (dilated)
        # Effect: Captures even larger spatial context at lower resolution
        self.dil_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.bn_dil2 = nn.BatchNorm2d(32)

        # Block 7: Downsampling and channel increase
        # Input: 32x16x16, Output: 64x8x8, RF: 3x3, stride=2
        # Effect: Reduces spatial size and increases feature channels for richer representation
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Block 8: Large receptive field with high dilation
        # Input: 64x8x8, Output: 64x8x8, RF: 11x11 (dilated)
        # Effect: Aggregates global context, helps with classification of large patterns
        self.dil_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=5, dilation=5)
        self.bn_dil3 = nn.BatchNorm2d(64)

        # Global average pooling and classifier
        # Effect: Reduces each feature map to a single value, then classifies
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Output: 64x1x1
        self.fc = nn.Linear(64, 10)  # Output: 10 classes

    def forward(self, x):
        # Forward pass through all blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn_sep1(self.sep_conv1(x)))
        x = F.relu(self.bn_dil1(self.dil_conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn_sep2(self.sep_conv2(x)))
        x = F.relu(self.bn_dil2(self.dil_conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn_dil3(self.dil_conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate model and print parameter count
model = Net()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training and validation metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
num_epochs = 40

# Training loop: trains model and prints batch/epoch stats
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).int().sum().item()
        # Print training stats every 100 batches
        if i % 100 == 99:
            print(
                f'Epoch {epoch + 1}, Batch {i + 1}: Train Loss: {running_loss / 100:.4f}, Train Accuracy: {100 * correct / total:.2f}%'
            )
            running_loss = 0.0
            correct = 0
            total = 0

    # Calculate epoch-wise training metrics
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).int().sum().item()
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation loop: evaluate on test set
    model.eval()
    correct = 0
    total = 0
    running_test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).int().sum().item()
    test_loss = running_test_loss / len(testloader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # Print epoch summary
    print(
        f'Epoch {epoch + 1}/{num_epochs}, '
        f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
        f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%'
    )

print('Finished Training')

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
