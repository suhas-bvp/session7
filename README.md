# Experiment 2 (experiment2_cifar_cnn.py)
  --> CIFAR-10 CNN Model with Albumentations Augmentation

## Model Overview
This project implements a Convolutional Neural Network (CNN) for CIFAR-10 image classification using PyTorch. The training pipeline leverages the Albumentations library for advanced data augmentation, including horizontal flip, affine transformations, and coarse dropout.

## Number of Parameters
The total number of trainable parameters in the model is **201,802**.

### Parameter Calculation Breakdown

| Layer         | Formula                                      | Parameters |
|-------------- |----------------------------------------------|------------|
| conv1         | (3×16×3×3) + 16 (bias)                       | 448        |
| bn1           | 16×2                                         | 32         |
| sep_conv1     | (16×3×3)+16 (bias) + (16×32×1×1)+32 (bias)   | 160+544=704|
| bn_sep1       | 32×2                                         | 64         |
| dil_conv1     | (32×32×3×3) + 32 (bias)                      | 9,248      |
| bn_dil1       | 32×2                                         | 64         |
| conv2         | (32×32×3×3) + 32 (bias)                      | 9,248      |
| bn2           | 32×2                                         | 64         |
| sep_conv2     | (32×3×3)+32 (bias) + (32×32×1×1)+32 (bias)   | 320+1,056=1,376|
| bn_sep2       | 32×2                                         | 64         |
| dil_conv2     | (32×32×3×3) + 32 (bias)                      | 9,248      |
| bn_dil2       | 32×2                                         | 64         |
| conv3         | (32×64×3×3) + 64 (bias)                      | 18,496     |
| bn3           | 64×2                                         | 128        |
| dil_conv3     | (64×64×3×3) + 64 (bias)                      | 36,928     |
| bn_dil3       | 64×2                                         | 128        |
| fc            | (64×10) + 10 (bias)                          | 650        |
| **Total**     |                                              | **201,802**|

## CNN Architecture
The model consists of the following blocks:

1. **Initial Feature Extraction**
   - `Conv2d(3, 16, 3x3, padding=1)` + `BatchNorm2d(16)`
   - Effect: Extracts low-level features (edges, colors) from the input image.

2. **Depthwise Separable Convolution**
   - `Conv2d(16, 16, 3x3, padding=1, groups=16)` (depthwise)
   - `Conv2d(16, 32, 1x1)` (pointwise)
   - `BatchNorm2d(32)`
   - Effect: Efficiently increases feature complexity while keeping parameter count low.

3. **Dilated Convolution**
   - `Conv2d(32, 32, 3x3, padding=2, dilation=2)` + `BatchNorm2d(32)`
   - Effect: Captures larger spatial context without increasing parameters.

4. **Downsampling**
   - `Conv2d(32, 32, 3x3, padding=1, stride=2)` + `BatchNorm2d(32)`
   - Effect: Reduces spatial resolution to focus on higher-level features.

5. **Depthwise Separable Convolution (Lower Resolution)**
   - `Conv2d(32, 32, 3x3, padding=1, groups=32)` (depthwise)
   - `Conv2d(32, 32, 1x1)` (pointwise)
   - `BatchNorm2d(32)`
   - Effect: Further increases feature complexity efficiently at lower resolution.

6. **Dilated Convolution (Lower Resolution)**
   - `Conv2d(32, 32, 3x3, padding=2, dilation=2)` + `BatchNorm2d(32)`
   - Effect: Captures even larger spatial context at lower resolution.

7. **Downsampling and Channel Increase**
   - `Conv2d(32, 64, 3x3, padding=1, stride=2)` + `BatchNorm2d(64)`
   - Effect: Reduces spatial size and increases feature channels for richer representation.

8. **Large Receptive Field with High Dilation**
   - `Conv2d(64, 64, 3x3, padding=5, dilation=5)` + `BatchNorm2d(64)`
   - Effect: Aggregates global context, helps with classification of large patterns.

9. **Global Average Pooling and Classifier**
   - `AdaptiveAvgPool2d((1, 1))`
   - `Linear(64, 10)`
   - Effect: Reduces each feature map to a single value, then classifies.


## Learning

- **Data Augmentation**: Techniques like horizontal flip, affine transformations, and coarse dropout increase the diversity of the training data, helping the model generalize better and reducing overfitting. This is especially important for small datasets like CIFAR-10, where augmentations simulate real-world variations and improve robustness.
  - **Horizontal Flip**: Randomly flips images left-right.
  - **Affine**: Randomly shifts, scales, and rotates images.
  - **CoarseDropout**: Randomly masks out a 16x16 region, filled with the dataset mean.
  - **Normalization**: Uses CIFAR-10 mean and std.

- **Depthwise Separable Convolution**: This operation factorizes a standard convolution into a depthwise (per-channel) convolution followed by a pointwise (1x1) convolution. It drastically reduces the number of parameters and computations, allowing the network to be deeper or wider without increasing model size, while still capturing complex features.

- **Dilated Convolution**: By introducing gaps (dilations) between kernel elements, dilated convolutions expand the receptive field without increasing the number of parameters or reducing spatial resolution. This helps the model capture more global context and relationships in the image, which is useful for recognizing larger patterns or objects.

- **Global Average Pooling (GAP) and Fully Connected (FC) Layer**: GAP reduces each feature map to a single value by averaging, which minimizes overfitting and enforces spatial invariance. The final FC layer then maps these global features to class scores. This combination is efficient and effective for classification tasks, as it reduces the number of parameters compared to flattening the entire feature map.

## Accuracy and Output

<!-- Add your accuracy and output results here -->



<img width="1435" height="865" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/10d710ae-ed3d-4153-b459-31f9bbe46dcf" />


<img width="1437" height="897" alt="image" src="https://github.com/user-attachments/assets/a85083f5-272b-40b6-979f-9c7240d7f88a" />


source "/Users/suhas/Desktop/ERA V4/session7/session7/.venv/bin/activate"
suhas@192 session7 % source "/Users/suhas/Desktop/ERA V4/session7/session7/.venv/bin/activate"
(.venv) suhas@192 session7 % "/Users/suhas/Desktop/ERA V4/session7/session7/.venv/bin/python" "/Users/suhas/Deskto
p/ERA V4/session7/session7/experiment2_cifar_cnn.py"
Files already downloaded and verified
Files already downloaded and verified
Total trainable parameters: 146570
Epoch 1, Batch 100: Train Loss: 1.8062, Train Accuracy: 32.94%
Epoch 1, Batch 200: Train Loss: 1.5193, Train Accuracy: 44.91%
Epoch 1, Batch 300: Train Loss: 1.3483, Train Accuracy: 51.47%
Epoch 1, Batch 400: Train Loss: 1.2630, Train Accuracy: 54.86%
Epoch 1, Batch 500: Train Loss: 1.1900, Train Accuracy: 56.83%
Epoch 1, Batch 600: Train Loss: 1.1221, Train Accuracy: 60.17%
Epoch 1, Batch 700: Train Loss: 1.0506, Train Accuracy: 62.06%
Epoch 1/20, Train Loss: 1.0236, Train Accuracy: 63.37%, Test Loss: 1.0360, Test Accuracy: 62.54%
Epoch 2, Batch 100: Train Loss: 0.9886, Train Accuracy: 65.16%
Epoch 2, Batch 200: Train Loss: 0.9707, Train Accuracy: 65.30%
Epoch 2, Batch 300: Train Loss: 0.9346, Train Accuracy: 67.14%
Epoch 2, Batch 400: Train Loss: 0.9429, Train Accuracy: 66.25%
Epoch 2, Batch 500: Train Loss: 0.9075, Train Accuracy: 67.88%
Epoch 2, Batch 600: Train Loss: 0.9047, Train Accuracy: 67.81%
Epoch 2, Batch 700: Train Loss: 0.8677, Train Accuracy: 69.55%
Epoch 2/20, Train Loss: 0.8007, Train Accuracy: 71.88%, Test Loss: 0.8456, Test Accuracy: 69.59%
Epoch 3, Batch 100: Train Loss: 0.8047, Train Accuracy: 71.98%
Epoch 3, Batch 200: Train Loss: 0.7937, Train Accuracy: 72.47%
Epoch 3, Batch 300: Train Loss: 0.8010, Train Accuracy: 72.31%
Epoch 3, Batch 400: Train Loss: 0.7874, Train Accuracy: 72.23%
Epoch 3, Batch 500: Train Loss: 0.7541, Train Accuracy: 73.19%
Epoch 3, Batch 600: Train Loss: 0.7553, Train Accuracy: 73.42%
Epoch 3, Batch 700: Train Loss: 0.7403, Train Accuracy: 73.92%
Epoch 3/20, Train Loss: 0.6752, Train Accuracy: 76.62%, Test Loss: 0.7377, Test Accuracy: 74.47%
Epoch 4, Batch 100: Train Loss: 0.6827, Train Accuracy: 75.97%
Epoch 4, Batch 200: Train Loss: 0.6933, Train Accuracy: 76.23%
Epoch 4, Batch 300: Train Loss: 0.6915, Train Accuracy: 75.38%
Epoch 4, Batch 400: Train Loss: 0.6769, Train Accuracy: 76.53%
Epoch 4, Batch 500: Train Loss: 0.6670, Train Accuracy: 76.73%
Epoch 4, Batch 600: Train Loss: 0.6913, Train Accuracy: 75.69%
Epoch 4, Batch 700: Train Loss: 0.6729, Train Accuracy: 76.48%
Epoch 4/20, Train Loss: 0.6210, Train Accuracy: 78.43%, Test Loss: 0.7000, Test Accuracy: 75.81%
Epoch 5, Batch 100: Train Loss: 0.6001, Train Accuracy: 78.80%
Epoch 5, Batch 200: Train Loss: 0.5919, Train Accuracy: 79.59%
Epoch 5, Batch 300: Train Loss: 0.6054, Train Accuracy: 78.25%
Epoch 5, Batch 400: Train Loss: 0.6092, Train Accuracy: 79.00%
Epoch 5, Batch 500: Train Loss: 0.6212, Train Accuracy: 78.22%
Epoch 5, Batch 600: Train Loss: 0.5971, Train Accuracy: 79.05%
Epoch 5, Batch 700: Train Loss: 0.6179, Train Accuracy: 78.34%
Epoch 5/20, Train Loss: 0.5305, Train Accuracy: 81.73%, Test Loss: 0.6474, Test Accuracy: 77.33%
Epoch 6, Batch 100: Train Loss: 0.5218, Train Accuracy: 82.42%
Epoch 6, Batch 200: Train Loss: 0.5509, Train Accuracy: 80.84%
Epoch 6, Batch 300: Train Loss: 0.5393, Train Accuracy: 81.69%
Epoch 6, Batch 400: Train Loss: 0.5563, Train Accuracy: 80.67%
Epoch 6, Batch 500: Train Loss: 0.5461, Train Accuracy: 81.06%
Epoch 6, Batch 600: Train Loss: 0.5414, Train Accuracy: 81.02%
Epoch 6, Batch 700: Train Loss: 0.5702, Train Accuracy: 79.91%
Epoch 6/20, Train Loss: 0.4849, Train Accuracy: 83.35%, Test Loss: 0.6412, Test Accuracy: 77.84%
Epoch 7, Batch 100: Train Loss: 0.4793, Train Accuracy: 83.67%
Epoch 7, Batch 200: Train Loss: 0.4917, Train Accuracy: 83.33%
Epoch 7, Batch 300: Train Loss: 0.4763, Train Accuracy: 83.75%
Epoch 7, Batch 400: Train Loss: 0.5097, Train Accuracy: 81.64%
Epoch 7, Batch 500: Train Loss: 0.5096, Train Accuracy: 81.94%
Epoch 7, Batch 600: Train Loss: 0.5048, Train Accuracy: 82.02%
Epoch 7, Batch 700: Train Loss: 0.5145, Train Accuracy: 82.62%
Epoch 7/20, Train Loss: 0.4195, Train Accuracy: 85.82%, Test Loss: 0.6015, Test Accuracy: 78.88%
Epoch 8, Batch 100: Train Loss: 0.4110, Train Accuracy: 86.14%
Epoch 8, Batch 200: Train Loss: 0.4388, Train Accuracy: 84.75%
Epoch 8, Batch 300: Train Loss: 0.4641, Train Accuracy: 84.05%
Epoch 8, Batch 400: Train Loss: 0.4591, Train Accuracy: 84.11%
Epoch 8, Batch 500: Train Loss: 0.4699, Train Accuracy: 83.70%
Epoch 8, Batch 600: Train Loss: 0.4749, Train Accuracy: 83.23%
Epoch 8, Batch 700: Train Loss: 0.4610, Train Accuracy: 83.48%
Epoch 8/20, Train Loss: 0.3851, Train Accuracy: 86.87%, Test Loss: 0.5948, Test Accuracy: 79.60%
Epoch 9, Batch 100: Train Loss: 0.4063, Train Accuracy: 86.06%
Epoch 9, Batch 200: Train Loss: 0.4062, Train Accuracy: 85.58%
Epoch 9, Batch 300: Train Loss: 0.4088, Train Accuracy: 85.64%
Epoch 9, Batch 400: Train Loss: 0.4266, Train Accuracy: 84.77%
Epoch 9, Batch 500: Train Loss: 0.4197, Train Accuracy: 85.38%
Epoch 9, Batch 600: Train Loss: 0.4259, Train Accuracy: 84.84%
Epoch 9, Batch 700: Train Loss: 0.4487, Train Accuracy: 84.45%
Epoch 9/20, Train Loss: 0.3520, Train Accuracy: 87.98%, Test Loss: 0.6014, Test Accuracy: 79.48%
Epoch 10, Batch 100: Train Loss: 0.3350, Train Accuracy: 88.78%
Epoch 10, Batch 200: Train Loss: 0.3656, Train Accuracy: 87.25%
Epoch 10, Batch 300: Train Loss: 0.3784, Train Accuracy: 87.16%
Epoch 10, Batch 400: Train Loss: 0.3879, Train Accuracy: 86.41%
Epoch 10, Batch 500: Train Loss: 0.3811, Train Accuracy: 86.47%
Epoch 10, Batch 600: Train Loss: 0.4107, Train Accuracy: 85.72%
Epoch 10, Batch 700: Train Loss: 0.4133, Train Accuracy: 85.45%
Epoch 10/20, Train Loss: 0.3273, Train Accuracy: 88.76%, Test Loss: 0.6117, Test Accuracy: 79.15%
Epoch 11, Batch 100: Train Loss: 0.3237, Train Accuracy: 88.80%
Epoch 11, Batch 200: Train Loss: 0.3393, Train Accuracy: 88.20%
Epoch 11, Batch 300: Train Loss: 0.3356, Train Accuracy: 88.34%
Epoch 11, Batch 400: Train Loss: 0.3405, Train Accuracy: 88.72%
Epoch 11, Batch 500: Train Loss: 0.3670, Train Accuracy: 86.83%
Epoch 11, Batch 600: Train Loss: 0.3846, Train Accuracy: 86.48%
Epoch 11, Batch 700: Train Loss: 0.3698, Train Accuracy: 87.17%
Epoch 11/20, Train Loss: 0.2839, Train Accuracy: 90.27%, Test Loss: 0.5982, Test Accuracy: 80.31%
Epoch 12, Batch 100: Train Loss: 0.2869, Train Accuracy: 90.22%
Epoch 12, Batch 200: Train Loss: 0.3026, Train Accuracy: 89.42%
Epoch 12, Batch 300: Train Loss: 0.3317, Train Accuracy: 88.30%
Epoch 12, Batch 400: Train Loss: 0.3346, Train Accuracy: 88.20%
Epoch 12, Batch 500: Train Loss: 0.3350, Train Accuracy: 88.16%
Epoch 12, Batch 600: Train Loss: 0.3419, Train Accuracy: 88.20%
Epoch 12, Batch 700: Train Loss: 0.3463, Train Accuracy: 87.75%





