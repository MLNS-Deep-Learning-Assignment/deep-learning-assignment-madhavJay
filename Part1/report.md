# Baseline Implementation
- Implemented a basic CNN with two convolution layers and a fully connected layer to do a Regression task. 
## Model Architecture

### Convolutional Layers
The convolutional section is responsible for feature extraction and consists of:
1. **Convolution Layer 1**: 
   - Input Channels: 1
   - Output Channels: 32
   - Kernel Size: 3x3
   - Padding: 1
   - Activation: ReLU
2. **MaxPooling Layer 1**: Reduces dimensions by a factor of 2.
3. **Convolution Layer 2**:
   - Input Channels: 32
   - Output Channels: 64
   - Kernel Size: 3x3
   - Padding: 1
   - Activation: ReLU
4. **MaxPooling Layer 2**: Further reduces dimensions by a factor of 2.

### Fully Connected Layers
This section processes the extracted features and makes the final prediction:
1. **Flatten Layer**: Converts the feature maps into a 1D vector.
2. **Fully Connected Layer 1**:
   - Input Features: 64 * 10 *42 (derived from the feature map dimensions after convolution and pooling)
   - Output Features: 128
   - Activation: ReLU
3. **Fully Connected Layer 2**:
   - Input Features: 128
   - Output Features: 1 (final prediction)

## Forward Pass
The forward pass consists of:
1. Passing the input through the convolutional layers for feature extraction.
2. Feeding the extracted features into the fully connected layers for regression.

## Other HyperParameters
- Learning Rate: 0.001
- Optimizer: Adam
- Criterion: MSELoss
- Epochs: 100
- Training/Val Split: 80-20

# Results of baseline
- Able to Adapt well to training data, ~0.15 MSE loss and 80% accuracy, however the validation loss is high and Accuracy is low (~12 and 11% respectively) Indicating the following:
    - High Training Accuracy: The model has learned the patterns in the training data very well, including noise or irrelevant features.

    - Low Validation Accuracy: The model struggles to perform well on new data because it hasn’t generalized the underlying patterns; instead, it has memorized the specifics of the training set.

# Some possible changes / improvements
- Data Augmentation
- Cross-Validation
- Regularization Techniques
- Ensemble Learning