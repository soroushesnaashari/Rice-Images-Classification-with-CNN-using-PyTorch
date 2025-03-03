## Rice Image Classification using CNN with PyTorch

### Overview
This project implements a Convolutional Neural Network (CNN) for classifying rice images into five categories (Arborio, Basmati, Ipsala, Jasmine, Karacadag) using PyTorch. The model is trained on 75,000 images (15,000 per class) with a focus on reproducibility, performance optimization, and modular code design. The custom CNN achieves **98% test accuracy**, leveraging PyTorch's flexibility for dynamic computation graphs and GPU acceleration.

<br>

### Project Workflow
1. **Data Preparation**:
   - Load dataset using PyTorch `ImageFolder` and split into training (80%), validation (10%), and test (10%) sets.
   - Apply transformations: Resize (224x224), normalization, and augmentation (random rotation, horizontal flip).
   - Create `DataLoader` instances with batch size 32 for efficient training.

2. **Model Architecture**:
   - **Custom CNN** (`RiceCNN` class):
     - Layers: `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d` → `Dropout` → Linear classifier.
     - Kernel sizes: 3x3 for convolutional layers, 2x2 for max pooling.

3. **Training Configuration**:
   - **Loss Function**: CrossEntropyLoss.
   - **Optimizer**: Adam with learning rate 0.001.
   - **Scheduler**: StepLR (gamma=0.1 every 7 epochs).
   - **Training Loop**: 15 epochs with batch-wise backpropagation and validation checks.

4. **Evaluation**:
   - Calculate test accuracy using `torchmetrics`.
   - Generate confusion matrix and classification report (precision, recall, F1-score).
   - Visualize training/validation loss curves and sample predictions.

5. **Inference**:
   - Save/load model weights (`rice_cnn.pth`) for predictions on new images.

<br>

### Key Features
- **PyTorch Flexibility**: Leverage dynamic computation graphs and GPU support via `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- **Data Pipeline**: Efficient loading with `DataLoader` and on-the-fly augmentation using `torchvision.transforms`.
- **Reproducibility**: Manual seed configuration (`torch.manual_seed(42)`).
- **Learning Rate Scheduling**: Adaptive LR adjustment for convergence optimization.
- **Metrics Tracking**: TensorBoard integration for loss/accuracy visualization (code-ready).

<br>

### Results
- **Test Accuracy**: **98%**  
- **Validation Accuracy**: Reached 97.5% by epoch 15  
- **Training Time**: ~30 seconds per epoch on GPU (Kaggle environment)  
- **Confusion Matrix**: Shows strong class separation with minor confusion between visually similar varieties (e.g., Jasmine vs. Basmati).  
- **Loss Curves**: Stable convergence without overfitting (training/validation loss: 0.06/0.08).

<br>

### Repository Contents
- **`rice-classification-cnn-using-pytorch.ipynb`**: Jupyter Notebook with full code, visualizations, and explanations.
- **`Data`:** Contains the [Original Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) and you can see the cleaned dataset in notebook.
- **`README.md`:** Project documentation.

<br>

### How to Contribute
Contributions are welcome! If you'd like to improve the project or add new features:

1. **Fork the repository.**
2. **Create a new branch.**
3. **Make your changes and submit a pull request.**
