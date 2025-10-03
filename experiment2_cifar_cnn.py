import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


batch_size = 100

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training and testing datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 3x32x32, Output: 16x32x32, RF: 3x3, Effect: Feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Depthwise Separable Conv Block 1
        # Input: 16x32x32, Output: 32x32x32, RF: 5x5, Effect: Feature extraction
        self.sep_conv1 = nn.Sequential(
            # Depthwise: Input: 16x32x32, Output: 16x32x32, RF: 5x5, Effect: Depthwise filtering
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            # Pointwise: Input: 16x32x32, Output: 32x32x32, RF: 5x5, Effect: Combining features
            nn.Conv2d(16, 32, kernel_size=1)
        )
        self.bn_sep1 = nn.BatchNorm2d(32)

        # Dilated Conv Block 1
        # Input: 32x32x32, Output: 32x32x32, RF: 9x9, Effect: Expanding receptive field without increasing parameters
        self.dil_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.bn_dil1 = nn.BatchNorm2d(32)

        # Downsampling using stride
        # Input: 32x32x32, Output: 64x16x16, RF: 11x11, Effect: Downsampling and feature extraction
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        # Depthwise Separable Conv Block 2
        # Input: 64x16x16, Output: 64x16x16, RF: 15x15, Effect: Feature extraction
        self.sep_conv2 = nn.Sequential(
            # Depthwise: Input: 64x16x16, Output: 64x16x16, RF: 15x15, Effect: Depthwise filtering
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            # Pointwise: Input: 64x16x16, Output: 64x16x16, RF: 15x15, Effect: Combining features
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.bn_sep2 = nn.BatchNorm2d(64)

        # Dilated Conv Block 2
        # Input: 64x16x16, Output: 64x16x16, RF: 23x23, Effect: Expanding receptive field without increasing parameters
        self.dil_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.bn_dil2 = nn.BatchNorm2d(64)

        # Downsampling using stride
        # Input: 64x16x16, Output: 128x8x8, RF: 27x27, Effect: Downsampling and feature extraction
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Average Pooling
        # Input: 128x8x8, Output: 128x1x1, RF: 27x27, Effect: Global feature aggregation
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        # Input: 128, Output: 10, RF: 27x27, Effect: Classification
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn_sep1(self.sep_conv1(x)))
        x = F.relu(self.bn_dil1(self.dil_conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn_sep2(self.sep_conv2(x)))
        x = F.relu(self.bn_dil2(self.dil_conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


model = Net()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


# 1. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Create lists to store training and validation loss and accuracy
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
num_epochs = 10

# 3. Implement the training loop
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    model.train()  # Set model to training mode
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print training statistics every 100 batches
        if i % 100 == 99:  # print every 100 mini-batches
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
            correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 4. Implement the validation loop
    model.eval()  # Set model to evaluation mode
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
            correct += (predicted == labels).sum().item()

    test_loss = running_test_loss / len(testloader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 5. Print the training and validation loss and accuracy after each epoch
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


