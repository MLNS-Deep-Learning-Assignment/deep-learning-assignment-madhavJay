# Baseline Implementation
- Implemented a basic CNN with two convolution layers and a fully connected layer to do a Regression task.

# Results of baseline
- Able to Adapt well to training data, ~0.15 MSE loss and 60% accuracy, however the validation loss is high and Accuracy is low (~12 and 11% respectively) Indicating the following:
    - High Training Accuracy: The model has learned the patterns in the training data very well, including noise or irrelevant features.

    - Low Validation Accuracy: The model struggles to perform well on new data because it hasn’t generalized the underlying patterns; instead, it has memorized the specifics of the training set.

# Some possible changes
- Data Augmentation
- Cross-Validation
- Regularization Techniques
- Ensemble Learning