# Enhanced Implementation

- Modified the baseline CNN by replacing it with a ResNet-based architecture for improved feature extraction and generalization in the regression task.

## Model Architecture

### ResNet Overview
ResNet, short for Residual Network, is a deep neural network architecture designed to address the problem of vanishing gradients in very deep networks. It introduces shortcut connections, or residual blocks, which allow the model to learn identity mappings. These shortcuts enable efficient training of very deep networks by preserving gradient flow, leading to improved accuracy and generalization. ResNet50, a variant with 50 layers, is widely used for feature extraction due to its pre-trained weights and ability to capture complex patterns in data.

### Feature Extraction (ResNet Backbone)
The feature extraction section uses a modified pre-trained ResNet50 to leverage its powerful feature extraction capabilities:

1. **Input Layer Adjustment**:
   - Original ResNet input layer modified to handle single-channel input:
     - Convolution Layer:
       - Input Channels: 1
       - Output Channels: 64
       - Kernel Size: 7x7
       - Stride: 2
       - Padding: 3
       - Bias: False
   - Pre-trained weights adjusted by summing across the input channel dimension.
2. **Feature Extraction Layers**:
   - ResNet50 backbone used for extracting features.
   - Fully connected output layer removed.

### Fully Connected Layers (MLP for Regression)
The fully connected section processes the extracted features to make the final prediction:

1. **MLP Layers**:
   - **Linear Layer 1**:
     - Input Features: 2048 (output size of ResNet feature extraction)
     - Output Features: 128
     - Activation: ReLU
   - **Linear Layer 2**:
     - Input Features: 128
     - Output Features: 1 (final regression output)

## Forward Pass
1. Input is passed through the modified ResNet50 for feature extraction.
2. Extracted features are flattened and passed through the MLP for regression.

## Other HyperParameters
- Learning Rate: 0.001
- Optimizer: Adam
- Criterion: MSELoss
- Epochs: 50
- Training/Val Split: 80-20

## Results of Enhanced Implementation
- **Training Results**: Achieved ~0.02 MSE loss and 99% accuracy, showing better adaptation to training data.
- **Validation Results**: Significant improvement with a validation MSE loss of ~0.39 and accuracy of 94%.

### Analysis:
- **Improved Feature Extraction**: The ResNet50 backbone provides better generalization due to pre-trained weights and deeper architecture.
- **Reduced Overfitting**: Enhanced performance on the validation set suggests better generalization compared to the baseline.

## Potential Further Improvements
1. **Data Augmentation**: Introduce transformations like rotation, scaling, or noise addition to increase training data diversity.
2. **Cross-Validation**: Use k-fold cross-validation to ensure robustness of results.
3. **Regularization Techniques**: Implement dropout or L2 regularization to reduce overfitting.
4. **Ensemble Learning**: Combine predictions from multiple models to improve accuracy further.
5. **Data Cleaning**: A lot of the images are very hard to make out the number, this might lead to overfitting.

## Code Implementation
```python
class DigitSumModel(nn.Module):
    def __init__(self):
        super(DigitSumModel, self).__init__()
        # Load a pre-trained ResNet and modify the input and output layers
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        state_dict = models.resnet50(pretrained=True).state_dict()
        state_dict['conv1.weight'] = state_dict['conv1.weight'].sum(dim=1, keepdim=True)
        self.resnet.load_state_dict(state_dict, strict=False)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # MLP to compute the sum of digits from extracted features
        self.mlp = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output single value for the sum
        )

    def forward(self, x):
        features = self.resnet(x)  # Extract features using ResNet
        output = self.mlp(features)  # Compute the sum using MLP
        return output
```

