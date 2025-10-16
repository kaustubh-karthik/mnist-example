# From-Scratch Neural Network Visualizer

This application demonstrates a simple neural network with no hidden layers (logistic regression) implemented entirely from scratch using only NumPy. It visualizes how the model processes MNIST digit images.

## Features

- **From-scratch implementation**: No PyTorch, TensorFlow, or other deep learning frameworks
- **Model persistence**: Automatically saves and loads trained models
- **Image navigation**: Browse through the MNIST dataset with Previous/Next buttons
- **Three visualization tabs**:
  1. **Original Image**: Shows the input image
  2. **Pixel Overlay**: Shows how each pixel contributes to the prediction
  3. **Weights & Analysis**: Shows model weights, probabilities, and feature importance

## Requirements

- Python 3.7+
- numpy
- matplotlib
- Pillow (PIL)
- tkinter (usually included with Python)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install numpy matplotlib Pillow
```

## Usage

1. Run the application:
```bash
python enhanced_nn_visualizer.py
```

2. The application will:
   - Check if a trained model exists (`trained_model.pkl`)
   - If not, train a new model on your MNIST dataset
   - Save the trained model for future use

3. Navigate through images:
   - Use "◀ Previous" and "Next ▶" buttons
   - View different visualizations in the tabs
   - See model predictions and confidence scores

## Model Architecture

- **Input**: 784 features (28×28 pixel image flattened)
- **Output**: 10 classes (digits 0-9)
- **Architecture**: Linear transformation + Softmax
- **Parameters**: 7,850 total (7,840 weights + 10 bias terms)

## Files

- `enhanced_nn_visualizer.py`: Main application
- `trained_model.pkl`: Saved model (created after first training)
- `mnist-dataset/`: Your MNIST dataset directory
- `img_1.jpg`: Your test image

## How It Works

1. **Forward Pass**: 
   - Linear transformation: `output = input @ weights.T + bias`
   - Softmax activation for probabilities

2. **Training**:
   - Cross-entropy loss function
   - Manual gradient computation
   - Stochastic gradient descent updates

3. **Visualization**:
   - Pixel overlay shows `image * weights` for the predicted class
   - Weight visualization shows learned patterns
   - Feature importance highlights most contributing pixels

The model learns to associate pixel patterns with digit classes through the weight matrix, which you can visualize to understand what the model "sees" when making predictions.

