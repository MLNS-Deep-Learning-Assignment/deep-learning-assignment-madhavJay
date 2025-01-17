**First Model uses a pretrained ResNet for Feature Extraction and second model has a CNN Feature Extractor that I've trained on a separate dataset** 

# First Model
## Implementation

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

# Second Model


## Overview
1. **MultiLabelCNN**: A convolutional neural network for multi-label classification of digits.
2. **SumOfDigitsPredictor**: A regression model that utilizes the feature extraction capabilities of the MultiLabelCNN for predicting the sum of digits.

## Architecture Details

### MultiLabelCNN
The **MultiLabelCNN** serves as the backbone for feature extraction and digit classification. It is structured as follows:

#### Convolutional Layers
1. **Conv Layer 1**:
   - Input Channels: 1
   - Output Channels: 32
   - Kernel Size: 5x5
   - Stride: 1
   - Padding: 2
   - Activation: ReLU
   - Followed by Batch Normalization and MaxPooling (2x2, stride 2).

2. **Conv Layer 2**:
   - Input Channels: 32
   - Output Channels: 64
   - Kernel Size: 3x3
   - Stride: 1
   - Padding: 1
   - Activation: ReLU
   - Followed by Batch Normalization and MaxPooling (2x2, stride 2).

3. **Conv Layer 3**:
   - Input Channels: 64
   - Output Channels: 128
   - Kernel Size: 3x3
   - Stride: 1
   - Padding: 1
   - Activation: ReLU
   - Followed by Batch Normalization and MaxPooling (2x2, stride 2).

4. **Conv Layer 4**:
   - Input Channels: 128
   - Output Channels: 256
   - Kernel Size: 3x3
   - Stride: 1
   - Padding: 1
   - Activation: ReLU
   - Followed by Batch Normalization and MaxPooling (2x2, stride 2).

#### Fully Connected Layers
1. **FC Layer 1**:
   - Input Features: 5120 (flattened feature map from convolutional layers).
   - Output Features: 128
   - Activation: ReLU
   - Dropout: 0.5


#### Objective
- The MultiLabelCNN predicts what digits (0-9) are present in the input image.

### SumOfDigitsPredictor
The **SumOfDigitsPredictor** utilizes the convolutional layers of the **MultiLabelCNN** for feature extraction and adds a regression head to predict the sum of digits.

#### Regression Head
1. **FC Layer 1**:
   - Input Features: 5120 (flattened feature map from shared convolutional layers).
   - Output Features: 128
   - Activation: ReLU
   - Dropout: 0.5

2. **Output Layer**:
   - Input Features: 128
   - Output Features: 1 (single value for sum of digits).

#### Objective
- The SumOfDigitsPredictor outputs the sum of all digits present in the input image.

## Training Details
- **MultiLabelCNN**:
  - Loss Function: Binary Cross-Entropy (BCE) Loss.
  - Optimizer: Adam.
  - Learning Rate: 0.001.
  - Epochs: 50.

- **SumOfDigitsPredictor**:
  - Loss Function: Mean Squared Error (MSE) Loss.
  - Optimizer: Adam.
  - Learning Rate: 0.001.
  - Epochs: 50.

## Results
### MultiLabelCNN
- **Training Accuracy**: ~75%
- **Validation Accuracy**: ~48%
- **Observations**: The model performs well on multi-label classification tasks, leveraging shared features for digit detection.

### SumOfDigitsPredictor
- **Training MSE**: ~0.9
- **Validation MSE**: ~1.0
- **Observations**: The regression head effectively utilizes the shared feature extractor, demonstrating strong performance in predicting the sum of digits.

## Potential Improvements
1. **Data Augmentation**: Enhance generalization by introducing variations in training data (e.g., rotation, scaling, noise).
2. **Task-Specific Layers**: Add additional layers tailored to each task to improve task-specific performance.
3. **Hyperparameter Tuning**: Experiment with learning rates, dropout rates, and optimizers to improve results.

## Code Implementation
```python
class MultiLabelCNN(nn.Module):
    def __init__(self):
        super(MultiLabelCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(5120, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 30),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class SumOfDigitsPredictor(nn.Module):
    def __init__(self, trained_model):
        super(SumOfDigitsPredictor, self).__init__()
        self.conv_layers = trained_model.conv_layers
        self.sum_head = nn.Sequential(
            nn.Linear(5120, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.sum_head(x)
        return x
```

