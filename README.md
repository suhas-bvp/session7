# Experiment 2 (experiment2_cifar_cnn.py)
  --> CIFAR-10 CNN Model with Albumentations Augmentation

## Model Overview
This project implements a Convolutional Neural Network (CNN) for CIFAR-10 image classification using PyTorch. The training pipeline leverages the Albumentations library for advanced data augmentation, including horizontal flip, affine transformations, and coarse dropout.


## Number of Parameters
The total number of trainable parameters in the model is **86,954**.

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
| **Total**     |                                              | **86,954**|


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
<img width="1011" height="738" alt="image" src="https://github.com/user-attachments/assets/8755126f-ecc8-4a7b-a98c-4f4cd352faf1" />


<img width="1424" height="772" alt="image" src="https://github.com/user-attachments/assets/2082769c-d624-4ef3-9352-7aa84572b9b9" />

<img width="1429" height="786" alt="image" src="https://github.com/user-attachments/assets/c6b55b6c-d209-4951-8da0-544902eb059b" />


<!-- Add your accuracy and output results here -->
<pre><code>
/Users/321351/PycharmProjects/PythonProject3/.venv/bin/python /Users/321351/PycharmProjects/PythonProject3/session7_exp1.py 
Total trainable parameters: 86954
Epoch 1, Batch 100: Train Loss: 1.8961, Train Accuracy: 30.67%
Epoch 1, Batch 200: Train Loss: 1.6376, Train Accuracy: 39.08%
Epoch 1, Batch 300: Train Loss: 1.5321, Train Accuracy: 43.27%
Epoch 1, Batch 400: Train Loss: 1.4587, Train Accuracy: 46.39%
Epoch 1, Batch 500: Train Loss: 1.4062, Train Accuracy: 49.02%
Epoch 1, Batch 600: Train Loss: 1.3245, Train Accuracy: 51.89%
Epoch 1, Batch 700: Train Loss: 1.2862, Train Accuracy: 53.09%
Epoch 1/40, Train Loss: 1.2243, Train Accuracy: 55.88%, Test Loss: 1.1497, Test Accuracy: 58.42%
Epoch 2, Batch 100: Train Loss: 1.2107, Train Accuracy: 57.44%
Epoch 2, Batch 200: Train Loss: 1.1558, Train Accuracy: 58.36%
Epoch 2, Batch 300: Train Loss: 1.1280, Train Accuracy: 59.16%
Epoch 2, Batch 400: Train Loss: 1.0929, Train Accuracy: 60.58%
Epoch 2, Batch 500: Train Loss: 1.0884, Train Accuracy: 61.31%
Epoch 2, Batch 600: Train Loss: 1.0481, Train Accuracy: 61.59%
Epoch 2, Batch 700: Train Loss: 1.0450, Train Accuracy: 62.61%
Epoch 2/40, Train Loss: 1.0233, Train Accuracy: 63.33%, Test Loss: 0.9488, Test Accuracy: 65.55%
Epoch 3, Batch 100: Train Loss: 1.0096, Train Accuracy: 64.67%
Epoch 3, Batch 200: Train Loss: 1.0096, Train Accuracy: 63.67%
Epoch 3, Batch 300: Train Loss: 0.9407, Train Accuracy: 66.34%
Epoch 3, Batch 400: Train Loss: 0.9652, Train Accuracy: 65.08%
Epoch 3, Batch 500: Train Loss: 0.9610, Train Accuracy: 65.33%
Epoch 3, Batch 600: Train Loss: 0.9295, Train Accuracy: 67.34%
Epoch 3, Batch 700: Train Loss: 0.9396, Train Accuracy: 66.30%
Epoch 3/40, Train Loss: 0.8888, Train Accuracy: 68.71%, Test Loss: 0.8150, Test Accuracy: 70.97%
Epoch 4, Batch 100: Train Loss: 0.9039, Train Accuracy: 67.27%
Epoch 4, Batch 200: Train Loss: 0.8920, Train Accuracy: 67.95%
Epoch 4, Batch 300: Train Loss: 0.8841, Train Accuracy: 68.41%
Epoch 4, Batch 400: Train Loss: 0.8761, Train Accuracy: 68.42%
Epoch 4, Batch 500: Train Loss: 0.8538, Train Accuracy: 70.00%
Epoch 4, Batch 600: Train Loss: 0.8576, Train Accuracy: 69.27%
Epoch 4, Batch 700: Train Loss: 0.8459, Train Accuracy: 70.66%
Epoch 4/40, Train Loss: 0.8251, Train Accuracy: 70.90%, Test Loss: 0.7609, Test Accuracy: 72.79%
Epoch 5, Batch 100: Train Loss: 0.8301, Train Accuracy: 70.53%
Epoch 5, Batch 200: Train Loss: 0.8073, Train Accuracy: 72.28%
Epoch 5, Batch 300: Train Loss: 0.8278, Train Accuracy: 70.97%
Epoch 5, Batch 400: Train Loss: 0.8232, Train Accuracy: 71.12%
Epoch 5, Batch 500: Train Loss: 0.7957, Train Accuracy: 72.03%
Epoch 5, Batch 600: Train Loss: 0.8061, Train Accuracy: 71.88%
Epoch 5, Batch 700: Train Loss: 0.7965, Train Accuracy: 71.58%
Epoch 5/40, Train Loss: 0.7657, Train Accuracy: 73.23%, Test Loss: 0.7091, Test Accuracy: 75.07%
Epoch 6, Batch 100: Train Loss: 0.7648, Train Accuracy: 72.92%
Epoch 6, Batch 200: Train Loss: 0.7690, Train Accuracy: 72.23%
Epoch 6, Batch 300: Train Loss: 0.7538, Train Accuracy: 72.75%
Epoch 6, Batch 400: Train Loss: 0.7766, Train Accuracy: 72.31%
Epoch 6, Batch 500: Train Loss: 0.7758, Train Accuracy: 72.78%
Epoch 6, Batch 600: Train Loss: 0.7522, Train Accuracy: 73.86%
Epoch 6, Batch 700: Train Loss: 0.7530, Train Accuracy: 73.69%
Epoch 6/40, Train Loss: 0.7216, Train Accuracy: 74.59%, Test Loss: 0.6828, Test Accuracy: 76.55%
Epoch 7, Batch 100: Train Loss: 0.7289, Train Accuracy: 74.28%
Epoch 7, Batch 200: Train Loss: 0.7146, Train Accuracy: 74.62%
Epoch 7, Batch 300: Train Loss: 0.7296, Train Accuracy: 74.44%
Epoch 7, Batch 400: Train Loss: 0.7466, Train Accuracy: 73.94%
Epoch 7, Batch 500: Train Loss: 0.7184, Train Accuracy: 74.62%
Epoch 7, Batch 600: Train Loss: 0.7134, Train Accuracy: 75.42%
Epoch 7, Batch 700: Train Loss: 0.7286, Train Accuracy: 74.94%
Epoch 7/40, Train Loss: 0.6845, Train Accuracy: 76.04%, Test Loss: 0.6369, Test Accuracy: 77.95%
Epoch 8, Batch 100: Train Loss: 0.6912, Train Accuracy: 76.03%
Epoch 8, Batch 200: Train Loss: 0.6927, Train Accuracy: 75.33%
Epoch 8, Batch 300: Train Loss: 0.7098, Train Accuracy: 75.31%
Epoch 8, Batch 400: Train Loss: 0.7147, Train Accuracy: 75.06%
Epoch 8, Batch 500: Train Loss: 0.6983, Train Accuracy: 75.64%
Epoch 8, Batch 600: Train Loss: 0.6778, Train Accuracy: 75.67%
Epoch 8, Batch 700: Train Loss: 0.6909, Train Accuracy: 75.75%
Epoch 8/40, Train Loss: 0.6752, Train Accuracy: 76.22%, Test Loss: 0.6406, Test Accuracy: 77.56%
Epoch 9, Batch 100: Train Loss: 0.6875, Train Accuracy: 76.48%
Epoch 9, Batch 200: Train Loss: 0.6753, Train Accuracy: 76.31%
Epoch 9, Batch 300: Train Loss: 0.6558, Train Accuracy: 77.31%
Epoch 9, Batch 400: Train Loss: 0.6781, Train Accuracy: 76.00%
Epoch 9, Batch 500: Train Loss: 0.6660, Train Accuracy: 76.45%
Epoch 9, Batch 600: Train Loss: 0.6678, Train Accuracy: 76.91%
Epoch 9, Batch 700: Train Loss: 0.6757, Train Accuracy: 76.86%
Epoch 9/40, Train Loss: 0.6609, Train Accuracy: 76.72%, Test Loss: 0.6237, Test Accuracy: 78.38%
Epoch 10, Batch 100: Train Loss: 0.6468, Train Accuracy: 76.98%
Epoch 10, Batch 200: Train Loss: 0.6362, Train Accuracy: 77.47%
Epoch 10, Batch 300: Train Loss: 0.6492, Train Accuracy: 77.45%
Epoch 10, Batch 400: Train Loss: 0.6422, Train Accuracy: 78.17%
Epoch 10, Batch 500: Train Loss: 0.6541, Train Accuracy: 77.58%
Epoch 10, Batch 600: Train Loss: 0.6405, Train Accuracy: 77.47%
Epoch 10, Batch 700: Train Loss: 0.6380, Train Accuracy: 77.73%
Epoch 10/40, Train Loss: 0.6291, Train Accuracy: 77.99%, Test Loss: 0.5925, Test Accuracy: 79.37%
Epoch 11, Batch 100: Train Loss: 0.6271, Train Accuracy: 78.00%
Epoch 11, Batch 200: Train Loss: 0.6343, Train Accuracy: 77.91%
Epoch 11, Batch 300: Train Loss: 0.6437, Train Accuracy: 77.45%
Epoch 11, Batch 400: Train Loss: 0.6349, Train Accuracy: 78.03%
Epoch 11, Batch 500: Train Loss: 0.6106, Train Accuracy: 79.02%
Epoch 11, Batch 600: Train Loss: 0.6138, Train Accuracy: 78.72%
Epoch 11, Batch 700: Train Loss: 0.6144, Train Accuracy: 78.89%
Epoch 11/40, Train Loss: 0.6159, Train Accuracy: 78.50%, Test Loss: 0.5836, Test Accuracy: 79.75%
Epoch 12, Batch 100: Train Loss: 0.5974, Train Accuracy: 79.34%
Epoch 12, Batch 200: Train Loss: 0.6049, Train Accuracy: 78.89%
Epoch 12, Batch 300: Train Loss: 0.6193, Train Accuracy: 78.06%
Epoch 12, Batch 400: Train Loss: 0.6283, Train Accuracy: 78.14%
Epoch 12, Batch 500: Train Loss: 0.6036, Train Accuracy: 79.12%
Epoch 12, Batch 600: Train Loss: 0.6116, Train Accuracy: 78.03%
Epoch 12, Batch 700: Train Loss: 0.5966, Train Accuracy: 78.58%
Epoch 12/40, Train Loss: 0.5839, Train Accuracy: 79.62%, Test Loss: 0.5562, Test Accuracy: 80.62%
Epoch 13, Batch 100: Train Loss: 0.5744, Train Accuracy: 80.31%
Epoch 13, Batch 200: Train Loss: 0.5806, Train Accuracy: 80.20%
Epoch 13, Batch 300: Train Loss: 0.6070, Train Accuracy: 78.28%
Epoch 13, Batch 400: Train Loss: 0.6039, Train Accuracy: 78.50%
Epoch 13, Batch 500: Train Loss: 0.5839, Train Accuracy: 79.95%
Epoch 13, Batch 600: Train Loss: 0.6073, Train Accuracy: 79.17%
Epoch 13, Batch 700: Train Loss: 0.5917, Train Accuracy: 79.03%
Epoch 13/40, Train Loss: 0.5772, Train Accuracy: 80.12%, Test Loss: 0.5521, Test Accuracy: 80.81%
Epoch 14, Batch 100: Train Loss: 0.5926, Train Accuracy: 78.98%
Epoch 14, Batch 200: Train Loss: 0.5962, Train Accuracy: 79.50%
Epoch 14, Batch 300: Train Loss: 0.5808, Train Accuracy: 80.39%
Epoch 14, Batch 400: Train Loss: 0.5875, Train Accuracy: 80.03%
Epoch 14, Batch 500: Train Loss: 0.5726, Train Accuracy: 79.88%
Epoch 14, Batch 600: Train Loss: 0.5868, Train Accuracy: 79.09%
Epoch 14, Batch 700: Train Loss: 0.5636, Train Accuracy: 80.48%
Epoch 14/40, Train Loss: 0.5732, Train Accuracy: 80.07%, Test Loss: 0.5552, Test Accuracy: 80.72%
Epoch 15, Batch 100: Train Loss: 0.5607, Train Accuracy: 80.84%
Epoch 15, Batch 200: Train Loss: 0.5827, Train Accuracy: 79.47%
Epoch 15, Batch 300: Train Loss: 0.5630, Train Accuracy: 80.66%
Epoch 15, Batch 400: Train Loss: 0.5583, Train Accuracy: 80.58%
Epoch 15, Batch 500: Train Loss: 0.5699, Train Accuracy: 80.19%
Epoch 15, Batch 600: Train Loss: 0.5660, Train Accuracy: 80.36%
Epoch 15, Batch 700: Train Loss: 0.5842, Train Accuracy: 79.61%
Epoch 15/40, Train Loss: 0.5630, Train Accuracy: 80.31%, Test Loss: 0.5572, Test Accuracy: 80.52%
Epoch 16, Batch 100: Train Loss: 0.5749, Train Accuracy: 80.41%
Epoch 16, Batch 200: Train Loss: 0.5558, Train Accuracy: 80.78%
Epoch 16, Batch 300: Train Loss: 0.5562, Train Accuracy: 80.42%
Epoch 16, Batch 400: Train Loss: 0.5438, Train Accuracy: 80.77%
Epoch 16, Batch 500: Train Loss: 0.5658, Train Accuracy: 79.80%
Epoch 16, Batch 600: Train Loss: 0.5452, Train Accuracy: 81.36%
Epoch 16, Batch 700: Train Loss: 0.5485, Train Accuracy: 80.53%
Epoch 16/40, Train Loss: 0.5473, Train Accuracy: 81.02%, Test Loss: 0.5396, Test Accuracy: 81.44%
Epoch 17, Batch 100: Train Loss: 0.5622, Train Accuracy: 79.86%
Epoch 17, Batch 200: Train Loss: 0.5617, Train Accuracy: 80.34%
Epoch 17, Batch 300: Train Loss: 0.5385, Train Accuracy: 81.45%
Epoch 17, Batch 400: Train Loss: 0.5191, Train Accuracy: 82.23%
Epoch 17, Batch 500: Train Loss: 0.5725, Train Accuracy: 79.81%
Epoch 17, Batch 600: Train Loss: 0.5558, Train Accuracy: 80.67%
Epoch 17, Batch 700: Train Loss: 0.5548, Train Accuracy: 80.56%
Epoch 17/40, Train Loss: 0.5363, Train Accuracy: 81.28%, Test Loss: 0.5419, Test Accuracy: 81.18%
Epoch 18, Batch 100: Train Loss: 0.5397, Train Accuracy: 81.12%
Epoch 18, Batch 200: Train Loss: 0.5268, Train Accuracy: 81.86%
Epoch 18, Batch 300: Train Loss: 0.5318, Train Accuracy: 81.30%
Epoch 18, Batch 400: Train Loss: 0.5484, Train Accuracy: 80.88%
Epoch 18, Batch 500: Train Loss: 0.5300, Train Accuracy: 81.98%
Epoch 18, Batch 600: Train Loss: 0.5529, Train Accuracy: 80.64%
Epoch 18, Batch 700: Train Loss: 0.5434, Train Accuracy: 81.22%
Epoch 18/40, Train Loss: 0.5183, Train Accuracy: 81.85%, Test Loss: 0.5204, Test Accuracy: 82.15%
Epoch 19, Batch 100: Train Loss: 0.5186, Train Accuracy: 81.98%
Epoch 19, Batch 200: Train Loss: 0.5472, Train Accuracy: 80.91%
Epoch 19, Batch 300: Train Loss: 0.5258, Train Accuracy: 81.72%
Epoch 19, Batch 400: Train Loss: 0.5437, Train Accuracy: 80.88%
Epoch 19, Batch 500: Train Loss: 0.5352, Train Accuracy: 81.55%
Epoch 19, Batch 600: Train Loss: 0.5161, Train Accuracy: 82.20%
Epoch 19, Batch 700: Train Loss: 0.5272, Train Accuracy: 81.50%
Epoch 19/40, Train Loss: 0.5088, Train Accuracy: 82.44%, Test Loss: 0.5066, Test Accuracy: 82.29%
Epoch 20, Batch 100: Train Loss: 0.4993, Train Accuracy: 82.38%
Epoch 20, Batch 200: Train Loss: 0.5285, Train Accuracy: 81.09%
Epoch 20, Batch 300: Train Loss: 0.5302, Train Accuracy: 81.77%
Epoch 20, Batch 400: Train Loss: 0.5299, Train Accuracy: 81.36%
Epoch 20, Batch 500: Train Loss: 0.5188, Train Accuracy: 81.47%
Epoch 20, Batch 600: Train Loss: 0.5239, Train Accuracy: 82.20%
Epoch 20, Batch 700: Train Loss: 0.5306, Train Accuracy: 81.69%
Epoch 20/40, Train Loss: 0.5026, Train Accuracy: 82.42%, Test Loss: 0.5050, Test Accuracy: 82.49%
Epoch 21, Batch 100: Train Loss: 0.5154, Train Accuracy: 81.73%
Epoch 21, Batch 200: Train Loss: 0.5173, Train Accuracy: 82.23%
Epoch 21, Batch 300: Train Loss: 0.5167, Train Accuracy: 81.98%
Epoch 21, Batch 400: Train Loss: 0.5244, Train Accuracy: 81.72%
Epoch 21, Batch 500: Train Loss: 0.5272, Train Accuracy: 81.23%
Epoch 21, Batch 600: Train Loss: 0.5239, Train Accuracy: 81.75%
Epoch 21, Batch 700: Train Loss: 0.5331, Train Accuracy: 81.56%
Epoch 21/40, Train Loss: 0.4982, Train Accuracy: 82.81%, Test Loss: 0.5044, Test Accuracy: 82.43%
Epoch 22, Batch 100: Train Loss: 0.5050, Train Accuracy: 82.23%
Epoch 22, Batch 200: Train Loss: 0.5038, Train Accuracy: 82.59%
Epoch 22, Batch 300: Train Loss: 0.4990, Train Accuracy: 82.44%
Epoch 22, Batch 400: Train Loss: 0.4940, Train Accuracy: 82.81%
Epoch 22, Batch 500: Train Loss: 0.5050, Train Accuracy: 82.28%
Epoch 22, Batch 600: Train Loss: 0.4967, Train Accuracy: 81.81%
Epoch 22, Batch 700: Train Loss: 0.5216, Train Accuracy: 82.17%
Epoch 22/40, Train Loss: 0.5002, Train Accuracy: 82.60%, Test Loss: 0.5117, Test Accuracy: 82.24%
Epoch 23, Batch 100: Train Loss: 0.5018, Train Accuracy: 82.38%
Epoch 23, Batch 200: Train Loss: 0.5026, Train Accuracy: 82.36%
Epoch 23, Batch 300: Train Loss: 0.5054, Train Accuracy: 82.39%
Epoch 23, Batch 400: Train Loss: 0.4845, Train Accuracy: 82.45%
Epoch 23, Batch 500: Train Loss: 0.4952, Train Accuracy: 82.83%
Epoch 23, Batch 600: Train Loss: 0.5000, Train Accuracy: 82.80%
Epoch 23, Batch 700: Train Loss: 0.5122, Train Accuracy: 81.88%
Epoch 23/40, Train Loss: 0.4835, Train Accuracy: 83.32%, Test Loss: 0.4871, Test Accuracy: 82.72%
Epoch 24, Batch 100: Train Loss: 0.4760, Train Accuracy: 83.17%
Epoch 24, Batch 200: Train Loss: 0.4777, Train Accuracy: 83.44%
Epoch 24, Batch 300: Train Loss: 0.4915, Train Accuracy: 82.86%
Epoch 24, Batch 400: Train Loss: 0.4956, Train Accuracy: 83.03%
Epoch 24, Batch 500: Train Loss: 0.4946, Train Accuracy: 82.86%
Epoch 24, Batch 600: Train Loss: 0.5015, Train Accuracy: 82.75%
Epoch 24, Batch 700: Train Loss: 0.4959, Train Accuracy: 82.83%
Epoch 24/40, Train Loss: 0.4827, Train Accuracy: 83.08%, Test Loss: 0.4909, Test Accuracy: 83.17%
Epoch 25, Batch 100: Train Loss: 0.4759, Train Accuracy: 83.75%
Epoch 25, Batch 200: Train Loss: 0.4737, Train Accuracy: 83.39%
Epoch 25, Batch 300: Train Loss: 0.4949, Train Accuracy: 82.70%
Epoch 25, Batch 400: Train Loss: 0.4926, Train Accuracy: 82.56%
Epoch 25, Batch 500: Train Loss: 0.4886, Train Accuracy: 82.84%
Epoch 25, Batch 600: Train Loss: 0.4888, Train Accuracy: 83.27%
Epoch 25, Batch 700: Train Loss: 0.4890, Train Accuracy: 83.08%
Epoch 25/40, Train Loss: 0.4732, Train Accuracy: 83.49%, Test Loss: 0.4779, Test Accuracy: 83.48%
Epoch 26, Batch 100: Train Loss: 0.4850, Train Accuracy: 83.45%
Epoch 26, Batch 200: Train Loss: 0.4796, Train Accuracy: 82.81%
Epoch 26, Batch 300: Train Loss: 0.4851, Train Accuracy: 82.94%
Epoch 26, Batch 400: Train Loss: 0.4802, Train Accuracy: 82.94%
Epoch 26, Batch 500: Train Loss: 0.4757, Train Accuracy: 83.20%
Epoch 26, Batch 600: Train Loss: 0.4747, Train Accuracy: 83.94%
Epoch 26, Batch 700: Train Loss: 0.5032, Train Accuracy: 82.36%
Epoch 26/40, Train Loss: 0.4708, Train Accuracy: 83.55%, Test Loss: 0.4876, Test Accuracy: 82.94%
Epoch 27, Batch 100: Train Loss: 0.4743, Train Accuracy: 84.16%
Epoch 27, Batch 200: Train Loss: 0.4599, Train Accuracy: 83.92%
Epoch 27, Batch 300: Train Loss: 0.4703, Train Accuracy: 83.66%
Epoch 27, Batch 400: Train Loss: 0.4601, Train Accuracy: 83.55%
Epoch 27, Batch 500: Train Loss: 0.4761, Train Accuracy: 83.50%
Epoch 27, Batch 600: Train Loss: 0.4636, Train Accuracy: 83.52%
Epoch 27, Batch 700: Train Loss: 0.4620, Train Accuracy: 83.53%
Epoch 27/40, Train Loss: 0.4555, Train Accuracy: 84.14%, Test Loss: 0.4771, Test Accuracy: 83.32%
Epoch 28, Batch 100: Train Loss: 0.4603, Train Accuracy: 83.98%
Epoch 28, Batch 200: Train Loss: 0.4599, Train Accuracy: 84.52%
Epoch 28, Batch 300: Train Loss: 0.4716, Train Accuracy: 84.02%
Epoch 28, Batch 400: Train Loss: 0.4700, Train Accuracy: 84.05%
Epoch 28, Batch 500: Train Loss: 0.4629, Train Accuracy: 84.02%
Epoch 28, Batch 600: Train Loss: 0.4874, Train Accuracy: 82.98%
Epoch 28, Batch 700: Train Loss: 0.4631, Train Accuracy: 83.86%
Epoch 28/40, Train Loss: 0.4467, Train Accuracy: 84.52%, Test Loss: 0.4793, Test Accuracy: 83.02%
Epoch 29, Batch 100: Train Loss: 0.4742, Train Accuracy: 83.36%
Epoch 29, Batch 200: Train Loss: 0.4439, Train Accuracy: 84.69%
Epoch 29, Batch 300: Train Loss: 0.4685, Train Accuracy: 83.48%
Epoch 29, Batch 400: Train Loss: 0.4728, Train Accuracy: 83.62%
Epoch 29, Batch 500: Train Loss: 0.4651, Train Accuracy: 84.11%
Epoch 29, Batch 600: Train Loss: 0.4619, Train Accuracy: 84.06%
Epoch 29, Batch 700: Train Loss: 0.4789, Train Accuracy: 83.66%
Epoch 29/40, Train Loss: 0.4489, Train Accuracy: 84.30%, Test Loss: 0.4844, Test Accuracy: 83.32%
Epoch 30, Batch 100: Train Loss: 0.4543, Train Accuracy: 84.19%
Epoch 30, Batch 200: Train Loss: 0.4675, Train Accuracy: 83.78%
Epoch 30, Batch 300: Train Loss: 0.4800, Train Accuracy: 83.39%
Epoch 30, Batch 400: Train Loss: 0.4598, Train Accuracy: 83.98%
Epoch 30, Batch 500: Train Loss: 0.4554, Train Accuracy: 84.00%
Epoch 30, Batch 600: Train Loss: 0.4567, Train Accuracy: 83.97%
Epoch 30, Batch 700: Train Loss: 0.4653, Train Accuracy: 83.81%
Epoch 30/40, Train Loss: 0.4456, Train Accuracy: 84.45%, Test Loss: 0.4645, Test Accuracy: 83.97%
Epoch 31, Batch 100: Train Loss: 0.4411, Train Accuracy: 84.78%
Epoch 31, Batch 200: Train Loss: 0.4537, Train Accuracy: 84.28%
Epoch 31, Batch 300: Train Loss: 0.4595, Train Accuracy: 83.41%
Epoch 31, Batch 400: Train Loss: 0.4477, Train Accuracy: 84.81%
Epoch 31, Batch 500: Train Loss: 0.4603, Train Accuracy: 84.17%
Epoch 31, Batch 600: Train Loss: 0.4579, Train Accuracy: 84.03%
Epoch 31, Batch 700: Train Loss: 0.4647, Train Accuracy: 83.36%
Epoch 31/40, Train Loss: 0.4406, Train Accuracy: 84.81%, Test Loss: 0.4657, Test Accuracy: 83.90%
Epoch 32, Batch 100: Train Loss: 0.4393, Train Accuracy: 84.70%
Epoch 32, Batch 200: Train Loss: 0.4417, Train Accuracy: 84.72%
Epoch 32, Batch 300: Train Loss: 0.4691, Train Accuracy: 84.36%
Epoch 32, Batch 400: Train Loss: 0.4506, Train Accuracy: 84.22%
Epoch 32, Batch 500: Train Loss: 0.4576, Train Accuracy: 83.81%
Epoch 32, Batch 600: Train Loss: 0.4628, Train Accuracy: 84.12%
Epoch 32, Batch 700: Train Loss: 0.4532, Train Accuracy: 84.48%
Epoch 32/40, Train Loss: 0.4321, Train Accuracy: 84.92%, Test Loss: 0.4616, Test Accuracy: 84.32%
Epoch 33, Batch 100: Train Loss: 0.4158, Train Accuracy: 85.62%
Epoch 33, Batch 200: Train Loss: 0.4573, Train Accuracy: 83.66%
Epoch 33, Batch 300: Train Loss: 0.4607, Train Accuracy: 83.95%
Epoch 33, Batch 400: Train Loss: 0.4536, Train Accuracy: 84.09%
Epoch 33, Batch 500: Train Loss: 0.4519, Train Accuracy: 84.27%
Epoch 33, Batch 600: Train Loss: 0.4315, Train Accuracy: 84.56%
Epoch 33, Batch 700: Train Loss: 0.4571, Train Accuracy: 84.27%
Epoch 33/40, Train Loss: 0.4216, Train Accuracy: 85.27%, Test Loss: 0.4577, Test Accuracy: 83.93%
Epoch 34, Batch 100: Train Loss: 0.4358, Train Accuracy: 84.66%
Epoch 34, Batch 200: Train Loss: 0.4518, Train Accuracy: 83.86%
Epoch 34, Batch 300: Train Loss: 0.4370, Train Accuracy: 84.89%
Epoch 34, Batch 400: Train Loss: 0.4542, Train Accuracy: 84.12%
Epoch 34, Batch 500: Train Loss: 0.4291, Train Accuracy: 85.39%
Epoch 34, Batch 600: Train Loss: 0.4292, Train Accuracy: 84.72%
Epoch 34, Batch 700: Train Loss: 0.4418, Train Accuracy: 84.69%
Epoch 34/40, Train Loss: 0.4245, Train Accuracy: 85.37%, Test Loss: 0.4644, Test Accuracy: 84.18%
Epoch 35, Batch 100: Train Loss: 0.4395, Train Accuracy: 84.58%
Epoch 35, Batch 200: Train Loss: 0.4371, Train Accuracy: 84.92%
Epoch 35, Batch 300: Train Loss: 0.4356, Train Accuracy: 84.33%
Epoch 35, Batch 400: Train Loss: 0.4236, Train Accuracy: 85.33%
Epoch 35, Batch 500: Train Loss: 0.4225, Train Accuracy: 85.28%
Epoch 35, Batch 600: Train Loss: 0.4384, Train Accuracy: 84.86%
Epoch 35, Batch 700: Train Loss: 0.4398, Train Accuracy: 85.27%
Epoch 35/40, Train Loss: 0.4219, Train Accuracy: 85.17%, Test Loss: 0.4671, Test Accuracy: 83.98%
Epoch 36, Batch 100: Train Loss: 0.4287, Train Accuracy: 84.73%
Epoch 36, Batch 200: Train Loss: 0.4268, Train Accuracy: 85.16%
Epoch 36, Batch 300: Train Loss: 0.4453, Train Accuracy: 83.81%
Epoch 36, Batch 400: Train Loss: 0.4475, Train Accuracy: 84.42%
Epoch 36, Batch 500: Train Loss: 0.4377, Train Accuracy: 85.11%
Epoch 36, Batch 600: Train Loss: 0.4283, Train Accuracy: 84.95%
Epoch 36, Batch 700: Train Loss: 0.4365, Train Accuracy: 84.75%
Epoch 36/40, Train Loss: 0.4142, Train Accuracy: 85.68%, Test Loss: 0.4515, Test Accuracy: 84.44%
Epoch 37, Batch 100: Train Loss: 0.4173, Train Accuracy: 85.53%
Epoch 37, Batch 200: Train Loss: 0.4195, Train Accuracy: 85.20%
Epoch 37, Batch 300: Train Loss: 0.4238, Train Accuracy: 85.22%
Epoch 37, Batch 400: Train Loss: 0.4384, Train Accuracy: 84.59%
Epoch 37, Batch 500: Train Loss: 0.4318, Train Accuracy: 85.16%
Epoch 37, Batch 600: Train Loss: 0.4252, Train Accuracy: 84.92%
Epoch 37, Batch 700: Train Loss: 0.4446, Train Accuracy: 84.38%
Epoch 37/40, Train Loss: 0.4132, Train Accuracy: 85.68%, Test Loss: 0.4542, Test Accuracy: 84.59%
Epoch 38, Batch 100: Train Loss: 0.4000, Train Accuracy: 85.64%
Epoch 38, Batch 200: Train Loss: 0.4121, Train Accuracy: 85.62%
Epoch 38, Batch 300: Train Loss: 0.4273, Train Accuracy: 85.23%
Epoch 38, Batch 400: Train Loss: 0.4402, Train Accuracy: 84.17%
Epoch 38, Batch 500: Train Loss: 0.4331, Train Accuracy: 84.94%
Epoch 38, Batch 600: Train Loss: 0.4297, Train Accuracy: 85.42%
Epoch 38, Batch 700: Train Loss: 0.4290, Train Accuracy: 85.31%
Epoch 38/40, Train Loss: 0.4203, Train Accuracy: 85.35%, Test Loss: 0.4646, Test Accuracy: 83.69%
Epoch 39, Batch 100: Train Loss: 0.4121, Train Accuracy: 86.08%
Epoch 39, Batch 200: Train Loss: 0.4073, Train Accuracy: 86.16%
Epoch 39, Batch 300: Train Loss: 0.4400, Train Accuracy: 84.64%
Epoch 39, Batch 400: Train Loss: 0.4378, Train Accuracy: 85.03%
Epoch 39, Batch 500: Train Loss: 0.4225, Train Accuracy: 85.17%
Epoch 39, Batch 600: Train Loss: 0.4094, Train Accuracy: 85.73%
Epoch 39, Batch 700: Train Loss: 0.4217, Train Accuracy: 85.31%
Epoch 39/40, Train Loss: 0.4066, Train Accuracy: 85.72%, Test Loss: 0.4501, Test Accuracy: 84.44%
Epoch 40, Batch 100: Train Loss: 0.4152, Train Accuracy: 85.62%
Epoch 40, Batch 200: Train Loss: 0.4101, Train Accuracy: 85.08%
Epoch 40, Batch 300: Train Loss: 0.4005, Train Accuracy: 86.22%
Epoch 40, Batch 400: Train Loss: 0.4259, Train Accuracy: 84.91%
Epoch 40, Batch 500: Train Loss: 0.4306, Train Accuracy: 84.59%
Epoch 40, Batch 600: Train Loss: 0.4191, Train Accuracy: 85.78%
Epoch 40, Batch 700: Train Loss: 0.4266, Train Accuracy: 85.23%
Epoch 40/40, Train Loss: 0.4127, Train Accuracy: 85.64%, Test Loss: 0.4636, Test Accuracy: 83.96%
Finished Training


  </code></pre>




