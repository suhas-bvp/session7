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
<img width="1341" height="1029" alt="image" src="https://github.com/user-attachments/assets/cd87c81d-ab0a-422a-970d-74b20003da27" />

<img width="1785" height="967" alt="image" src="https://github.com/user-attachments/assets/2a6f2d35-415f-4ed8-a58d-43d6566c94be" />
<img width="1823" height="991" alt="image" src="https://github.com/user-attachments/assets/474e57f6-9c06-4584-bff5-b2908027db74" />


<!-- Add your accuracy and output results here -->
<pre><code>
/Users/321351/PycharmProjects/PythonProject3/.venv/bin/python /Users/321351/PycharmProjects/PythonProject3/session7_exp1.py 
/Users/321351/PycharmProjects/PythonProject3/session7_exp1.py:32: UserWarning: Argument(s) 'max_holes, max_height, max_width, fill_value' are not valid for transform CoarseDropout
  A.CoarseDropout(
Total trainable parameters: 86954
Epoch 1, Batch 100: Train Loss: 1.8682, Train Accuracy: 32.52%
Epoch 1, Batch 200: Train Loss: 1.5774, Train Accuracy: 42.16%
Epoch 1, Batch 300: Train Loss: 1.4153, Train Accuracy: 47.95%
Epoch 1, Batch 400: Train Loss: 1.3559, Train Accuracy: 50.05%
Epoch 1, Batch 500: Train Loss: 1.2541, Train Accuracy: 54.22%
Epoch 1/30, Train Loss: 1.2246, Train Accuracy: 55.48%, Test Loss: 1.1432, Test Accuracy: 59.00%
Epoch 2, Batch 100: Train Loss: 1.2049, Train Accuracy: 56.06%
Epoch 2, Batch 200: Train Loss: 1.1368, Train Accuracy: 58.59%
Epoch 2, Batch 300: Train Loss: 1.0972, Train Accuracy: 59.87%
Epoch 2, Batch 400: Train Loss: 1.0651, Train Accuracy: 61.60%
Epoch 2, Batch 500: Train Loss: 1.0408, Train Accuracy: 62.73%
Epoch 2/30, Train Loss: 0.9952, Train Accuracy: 64.40%, Test Loss: 0.9283, Test Accuracy: 66.91%
Epoch 3, Batch 100: Train Loss: 0.9766, Train Accuracy: 65.04%
Epoch 3, Batch 200: Train Loss: 0.9755, Train Accuracy: 64.94%
Epoch 3, Batch 300: Train Loss: 0.9614, Train Accuracy: 65.32%
Epoch 3, Batch 400: Train Loss: 0.9387, Train Accuracy: 66.91%
Epoch 3, Batch 500: Train Loss: 0.9444, Train Accuracy: 66.61%
Epoch 3/30, Train Loss: 0.8893, Train Accuracy: 68.32%, Test Loss: 0.8377, Test Accuracy: 69.97%
Epoch 4, Batch 100: Train Loss: 0.8680, Train Accuracy: 69.05%
Epoch 4, Batch 200: Train Loss: 0.8804, Train Accuracy: 69.07%
Epoch 4, Batch 300: Train Loss: 0.8873, Train Accuracy: 68.08%
Epoch 4, Batch 400: Train Loss: 0.8507, Train Accuracy: 69.67%
Epoch 4, Batch 500: Train Loss: 0.8318, Train Accuracy: 70.32%
Epoch 4/30, Train Loss: 0.8161, Train Accuracy: 71.18%, Test Loss: 0.7593, Test Accuracy: 73.05%
Epoch 5, Batch 100: Train Loss: 0.8174, Train Accuracy: 71.07%
Epoch 5, Batch 200: Train Loss: 0.8191, Train Accuracy: 70.72%
Epoch 5, Batch 300: Train Loss: 0.8029, Train Accuracy: 71.54%
Epoch 5, Batch 400: Train Loss: 0.7891, Train Accuracy: 72.55%
Epoch 5, Batch 500: Train Loss: 0.7782, Train Accuracy: 72.55%
Epoch 5/30, Train Loss: 0.7746, Train Accuracy: 72.63%, Test Loss: 0.7421, Test Accuracy: 73.88%
Epoch 6, Batch 100: Train Loss: 0.7616, Train Accuracy: 73.00%
Epoch 6, Batch 200: Train Loss: 0.7591, Train Accuracy: 73.31%
Epoch 6, Batch 300: Train Loss: 0.7513, Train Accuracy: 73.39%
Epoch 6, Batch 400: Train Loss: 0.7665, Train Accuracy: 72.62%
Epoch 6, Batch 500: Train Loss: 0.7473, Train Accuracy: 73.19%
Epoch 6/30, Train Loss: 0.7211, Train Accuracy: 74.64%, Test Loss: 0.6773, Test Accuracy: 76.40%
Epoch 7, Batch 100: Train Loss: 0.7233, Train Accuracy: 75.11%
Epoch 7, Batch 200: Train Loss: 0.7100, Train Accuracy: 74.42%
Epoch 7, Batch 300: Train Loss: 0.7086, Train Accuracy: 75.05%
Epoch 7, Batch 400: Train Loss: 0.6995, Train Accuracy: 75.48%
Epoch 7, Batch 500: Train Loss: 0.7354, Train Accuracy: 74.06%
Epoch 7/30, Train Loss: 0.6905, Train Accuracy: 75.97%, Test Loss: 0.6614, Test Accuracy: 76.87%
Epoch 8, Batch 100: Train Loss: 0.7072, Train Accuracy: 75.21%
Epoch 8, Batch 200: Train Loss: 0.6754, Train Accuracy: 75.99%
Epoch 8, Batch 300: Train Loss: 0.6791, Train Accuracy: 76.05%
Epoch 8, Batch 400: Train Loss: 0.6750, Train Accuracy: 76.39%
Epoch 8, Batch 500: Train Loss: 0.6937, Train Accuracy: 75.37%
Epoch 8/30, Train Loss: 0.6653, Train Accuracy: 76.69%, Test Loss: 0.6208, Test Accuracy: 78.45%
Epoch 9, Batch 100: Train Loss: 0.6549, Train Accuracy: 76.72%
Epoch 9, Batch 200: Train Loss: 0.6725, Train Accuracy: 76.37%
Epoch 9, Batch 300: Train Loss: 0.6645, Train Accuracy: 76.25%
Epoch 9, Batch 400: Train Loss: 0.6772, Train Accuracy: 76.01%
Epoch 9, Batch 500: Train Loss: 0.6497, Train Accuracy: 77.85%
Epoch 9/30, Train Loss: 0.6291, Train Accuracy: 78.08%, Test Loss: 0.5991, Test Accuracy: 79.41%
Epoch 10, Batch 100: Train Loss: 0.6244, Train Accuracy: 78.32%
Epoch 10, Batch 200: Train Loss: 0.6517, Train Accuracy: 77.08%
Epoch 10, Batch 300: Train Loss: 0.6538, Train Accuracy: 76.90%
Epoch 10, Batch 400: Train Loss: 0.6369, Train Accuracy: 77.76%
Epoch 10, Batch 500: Train Loss: 0.6382, Train Accuracy: 77.61%
Epoch 10/30, Train Loss: 0.6294, Train Accuracy: 77.86%, Test Loss: 0.6103, Test Accuracy: 79.04%
Epoch 11, Batch 100: Train Loss: 0.6285, Train Accuracy: 77.97%
Epoch 11, Batch 200: Train Loss: 0.6538, Train Accuracy: 77.42%
Epoch 11, Batch 300: Train Loss: 0.6071, Train Accuracy: 78.50%
Epoch 11, Batch 400: Train Loss: 0.6193, Train Accuracy: 78.50%
Epoch 11, Batch 500: Train Loss: 0.6063, Train Accuracy: 78.91%
Epoch 11/30, Train Loss: 0.5987, Train Accuracy: 79.02%, Test Loss: 0.5867, Test Accuracy: 79.72%
Epoch 12, Batch 100: Train Loss: 0.6069, Train Accuracy: 78.73%
Epoch 12, Batch 200: Train Loss: 0.5971, Train Accuracy: 79.48%
Epoch 12, Batch 300: Train Loss: 0.5997, Train Accuracy: 79.12%
Epoch 12, Batch 400: Train Loss: 0.6025, Train Accuracy: 78.72%
Epoch 12, Batch 500: Train Loss: 0.6026, Train Accuracy: 79.32%
Epoch 12/30, Train Loss: 0.5923, Train Accuracy: 79.54%, Test Loss: 0.5699, Test Accuracy: 80.26%
Epoch 13, Batch 100: Train Loss: 0.5940, Train Accuracy: 79.29%
Epoch 13, Batch 200: Train Loss: 0.5873, Train Accuracy: 79.89%
Epoch 13, Batch 300: Train Loss: 0.5890, Train Accuracy: 79.36%
Epoch 13, Batch 400: Train Loss: 0.5938, Train Accuracy: 79.47%
Epoch 13, Batch 500: Train Loss: 0.6028, Train Accuracy: 78.43%
Epoch 13/30, Train Loss: 0.5748, Train Accuracy: 80.13%, Test Loss: 0.5528, Test Accuracy: 81.08%
Epoch 14, Batch 100: Train Loss: 0.5647, Train Accuracy: 80.68%
Epoch 14, Batch 200: Train Loss: 0.5787, Train Accuracy: 80.27%
Epoch 14, Batch 300: Train Loss: 0.5986, Train Accuracy: 78.89%
Epoch 14, Batch 400: Train Loss: 0.5786, Train Accuracy: 79.88%
Epoch 14, Batch 500: Train Loss: 0.5775, Train Accuracy: 79.62%
Epoch 14/30, Train Loss: 0.5526, Train Accuracy: 80.79%, Test Loss: 0.5392, Test Accuracy: 81.77%
Epoch 15, Batch 100: Train Loss: 0.5647, Train Accuracy: 80.43%
Epoch 15, Batch 200: Train Loss: 0.5607, Train Accuracy: 80.56%
Epoch 15, Batch 300: Train Loss: 0.5628, Train Accuracy: 80.40%
Epoch 15, Batch 400: Train Loss: 0.5587, Train Accuracy: 80.38%
Epoch 15, Batch 500: Train Loss: 0.5600, Train Accuracy: 80.61%
Epoch 15/30, Train Loss: 0.5377, Train Accuracy: 81.27%, Test Loss: 0.5350, Test Accuracy: 81.63%
Epoch 16, Batch 100: Train Loss: 0.5265, Train Accuracy: 81.54%
Epoch 16, Batch 200: Train Loss: 0.5604, Train Accuracy: 80.66%
Epoch 16, Batch 300: Train Loss: 0.5810, Train Accuracy: 79.75%
Epoch 16, Batch 400: Train Loss: 0.5317, Train Accuracy: 81.42%
Epoch 16, Batch 500: Train Loss: 0.5675, Train Accuracy: 80.61%
Epoch 16/30, Train Loss: 0.5516, Train Accuracy: 80.95%, Test Loss: 0.5502, Test Accuracy: 81.09%
Epoch 17, Batch 100: Train Loss: 0.5400, Train Accuracy: 81.24%
Epoch 17, Batch 200: Train Loss: 0.5547, Train Accuracy: 80.62%
Epoch 17, Batch 300: Train Loss: 0.5311, Train Accuracy: 81.64%
Epoch 17, Batch 400: Train Loss: 0.5479, Train Accuracy: 80.71%
Epoch 17, Batch 500: Train Loss: 0.5434, Train Accuracy: 81.04%
Epoch 17/30, Train Loss: 0.5267, Train Accuracy: 81.56%, Test Loss: 0.5243, Test Accuracy: 82.00%
Epoch 18, Batch 100: Train Loss: 0.5224, Train Accuracy: 81.76%
Epoch 18, Batch 200: Train Loss: 0.5303, Train Accuracy: 81.28%
Epoch 18, Batch 300: Train Loss: 0.5358, Train Accuracy: 81.30%
Epoch 18, Batch 400: Train Loss: 0.5236, Train Accuracy: 81.34%
Epoch 18, Batch 500: Train Loss: 0.5454, Train Accuracy: 81.18%
Epoch 18/30, Train Loss: 0.5170, Train Accuracy: 81.99%, Test Loss: 0.5177, Test Accuracy: 82.38%
Epoch 19, Batch 100: Train Loss: 0.5077, Train Accuracy: 82.41%
Epoch 19, Batch 200: Train Loss: 0.5211, Train Accuracy: 81.52%
Epoch 19, Batch 300: Train Loss: 0.5362, Train Accuracy: 80.87%
Epoch 19, Batch 400: Train Loss: 0.5362, Train Accuracy: 81.48%
Epoch 19, Batch 500: Train Loss: 0.5207, Train Accuracy: 81.89%
Epoch 19/30, Train Loss: 0.5038, Train Accuracy: 82.43%, Test Loss: 0.5201, Test Accuracy: 82.39%
Epoch 20, Batch 100: Train Loss: 0.5190, Train Accuracy: 81.97%
Epoch 20, Batch 200: Train Loss: 0.5072, Train Accuracy: 82.12%
Epoch 20, Batch 300: Train Loss: 0.5174, Train Accuracy: 81.92%
Epoch 20, Batch 400: Train Loss: 0.5338, Train Accuracy: 81.72%
Epoch 20, Batch 500: Train Loss: 0.5237, Train Accuracy: 81.91%
Epoch 20/30, Train Loss: 0.4969, Train Accuracy: 82.84%, Test Loss: 0.4987, Test Accuracy: 82.74%
Epoch 21, Batch 100: Train Loss: 0.5034, Train Accuracy: 82.20%
Epoch 21, Batch 200: Train Loss: 0.5075, Train Accuracy: 81.99%
Epoch 21, Batch 300: Train Loss: 0.5038, Train Accuracy: 82.31%
Epoch 21, Batch 400: Train Loss: 0.4914, Train Accuracy: 83.04%
Epoch 21, Batch 500: Train Loss: 0.5180, Train Accuracy: 81.59%
Epoch 21/30, Train Loss: 0.4909, Train Accuracy: 82.70%, Test Loss: 0.5038, Test Accuracy: 82.66%
Epoch 22, Batch 100: Train Loss: 0.5073, Train Accuracy: 82.82%
Epoch 22, Batch 200: Train Loss: 0.4968, Train Accuracy: 82.42%
Epoch 22, Batch 300: Train Loss: 0.4909, Train Accuracy: 82.59%
Epoch 22, Batch 400: Train Loss: 0.4980, Train Accuracy: 82.38%
Epoch 22, Batch 500: Train Loss: 0.5005, Train Accuracy: 82.75%
Epoch 22/30, Train Loss: 0.4879, Train Accuracy: 83.00%, Test Loss: 0.5027, Test Accuracy: 83.05%
Epoch 23, Batch 100: Train Loss: 0.4924, Train Accuracy: 83.05%
Epoch 23, Batch 200: Train Loss: 0.4923, Train Accuracy: 82.98%
Epoch 23, Batch 300: Train Loss: 0.4835, Train Accuracy: 83.06%
Epoch 23, Batch 400: Train Loss: 0.4925, Train Accuracy: 82.97%
Epoch 23, Batch 500: Train Loss: 0.4979, Train Accuracy: 82.74%
Epoch 23/30, Train Loss: 0.4885, Train Accuracy: 82.91%, Test Loss: 0.5014, Test Accuracy: 82.72%
Epoch 24, Batch 100: Train Loss: 0.4778, Train Accuracy: 83.37%
Epoch 24, Batch 200: Train Loss: 0.4962, Train Accuracy: 83.04%
Epoch 24, Batch 300: Train Loss: 0.4792, Train Accuracy: 83.18%
Epoch 24, Batch 400: Train Loss: 0.4900, Train Accuracy: 83.16%
Epoch 24, Batch 500: Train Loss: 0.4826, Train Accuracy: 83.44%
Epoch 24/30, Train Loss: 0.4616, Train Accuracy: 83.69%, Test Loss: 0.4781, Test Accuracy: 83.29%
Epoch 25, Batch 100: Train Loss: 0.4619, Train Accuracy: 83.92%
Epoch 25, Batch 200: Train Loss: 0.4921, Train Accuracy: 83.19%
Epoch 25, Batch 300: Train Loss: 0.4839, Train Accuracy: 83.70%
Epoch 25, Batch 400: Train Loss: 0.4771, Train Accuracy: 83.93%
Epoch 25, Batch 500: Train Loss: 0.4883, Train Accuracy: 83.00%
Epoch 25/30, Train Loss: 0.4625, Train Accuracy: 83.93%, Test Loss: 0.4857, Test Accuracy: 83.70%
Epoch 26, Batch 100: Train Loss: 0.4701, Train Accuracy: 83.62%
Epoch 26, Batch 200: Train Loss: 0.4682, Train Accuracy: 83.58%
Epoch 26, Batch 300: Train Loss: 0.4876, Train Accuracy: 83.23%
Epoch 26, Batch 400: Train Loss: 0.4660, Train Accuracy: 83.71%
Epoch 26, Batch 500: Train Loss: 0.4832, Train Accuracy: 83.28%
Epoch 26/30, Train Loss: 0.4637, Train Accuracy: 83.84%, Test Loss: 0.4797, Test Accuracy: 83.71%
Epoch 27, Batch 100: Train Loss: 0.4564, Train Accuracy: 83.83%
Epoch 27, Batch 200: Train Loss: 0.4766, Train Accuracy: 83.25%
Epoch 27, Batch 300: Train Loss: 0.4593, Train Accuracy: 83.98%
Epoch 27, Batch 400: Train Loss: 0.4729, Train Accuracy: 83.62%
Epoch 27, Batch 500: Train Loss: 0.4900, Train Accuracy: 82.60%
Epoch 27/30, Train Loss: 0.4488, Train Accuracy: 84.49%, Test Loss: 0.4811, Test Accuracy: 83.30%
Epoch 28, Batch 100: Train Loss: 0.4705, Train Accuracy: 83.52%
Epoch 28, Batch 200: Train Loss: 0.4595, Train Accuracy: 83.98%
Epoch 28, Batch 300: Train Loss: 0.4669, Train Accuracy: 83.62%
Epoch 28, Batch 400: Train Loss: 0.4531, Train Accuracy: 84.39%
Epoch 28, Batch 500: Train Loss: 0.4706, Train Accuracy: 83.49%
Epoch 28/30, Train Loss: 0.4532, Train Accuracy: 84.27%, Test Loss: 0.4770, Test Accuracy: 83.89%
Epoch 29, Batch 100: Train Loss: 0.4593, Train Accuracy: 83.66%
Epoch 29, Batch 200: Train Loss: 0.4579, Train Accuracy: 84.02%
Epoch 29, Batch 300: Train Loss: 0.4603, Train Accuracy: 83.75%
Epoch 29, Batch 400: Train Loss: 0.4621, Train Accuracy: 83.67%
Epoch 29, Batch 500: Train Loss: 0.4542, Train Accuracy: 84.18%
Epoch 29/30, Train Loss: 0.4487, Train Accuracy: 84.27%, Test Loss: 0.4740, Test Accuracy: 83.77%
Epoch 30, Batch 100: Train Loss: 0.4384, Train Accuracy: 84.73%
Epoch 30, Batch 200: Train Loss: 0.4552, Train Accuracy: 84.04%
Epoch 30, Batch 300: Train Loss: 0.4593, Train Accuracy: 83.75%
Epoch 30, Batch 400: Train Loss: 0.4629, Train Accuracy: 83.88%
Epoch 30, Batch 500: Train Loss: 0.4532, Train Accuracy: 84.59%
Epoch 30/30, Train Loss: 0.4530, Train Accuracy: 84.26%, Test Loss: 0.4712, Test Accuracy: 84.11%
Finished Training

  </code></pre>




