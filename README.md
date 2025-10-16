# From-Scratch Neural Network Visualizer

This application demonstrates a simple neural network with no hidden layers (logistic regression) implemented entirely from scratch using only NumPy. It visualizes how the model processes MNIST digit images.

## Features

- **From-scratch implementation**: No PyTorch, TensorFlow, or other deep learning frameworks
- **Model persistence**: Automatically saves and loads trained models
- **Image navigation**: Browse through the MNIST dataset with Previous/Next buttons
- **Model improvement**: "Improve Model" button for advanced training with learning rate decay and early stopping
- **Three visualization tabs**:
  1. **Original Image**: Shows the input image
  2. **Pixel Overlay**: Shows how each pixel contributes to the prediction
  3. **Weights & Analysis**: Shows model weights, probabilities, and feature importance

## Project Structure

The code is organized into separate modules for better maintainability:

- `main.py`: Main application entry point
- `model.py`: Neural network model implementation
- `data_loader.py`: MNIST data loading and preprocessing
- `ui_components.py`: User interface components and visualizations
- `requirements.txt`: Python dependencies
- `trained_model.pkl`: Saved model (created after first training)
- `mnist-dataset/`: MNIST dataset directory
- `img_1.jpg`: Test image

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
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. The application will:
   - Check if a trained model exists (`trained_model.pkl`)
   - If not, train a new model on your MNIST dataset
   - Save the trained model for future use

3. Navigate through images:
   - Use "◀ Previous" and "Next ▶" buttons
   - View different visualizations in the tabs
   - See model predictions and confidence scores

4. Improve the model:
   - Click "Improve Model" for advanced training
   - Uses learning rate decay and early stopping
   - Shows performance metrics after training

## Model Architecture

- **Input**: 784 features (28×28 pixel image flattened)
- **Output**: 10 classes (digits 0-9)
- **Architecture**: Linear transformation + Softmax
- **Parameters**: 7,850 total (7,840 weights + 10 bias terms)

## Code Organization

### `model.py`
- `SimpleNeuralNetwork`: Core neural network implementation
- Forward/backward pass, training, prediction methods
- Model saving/loading functionality

### `data_loader.py`
- `MNISTDataLoader`: Handles loading and preprocessing MNIST images
- Label mapping and data splitting
- Single image loading for visualization

### `ui_components.py`
- `NNVisualizerUI`: All user interface components
- Visualization tabs and navigation controls
- Matplotlib integration and figure management

### `main.py`
- `NNVisualizerApp`: Main application coordinator
- Handles callbacks between UI and model
- Manages training and model improvement workflows

## How It Works

1. **Forward Pass**: 
   - Linear transformation: `output = input @ weights.T + bias`
   - Softmax activation for probabilities

2. **Training**:
   - Cross-entropy loss function
   - Manual gradient computation
   - Stochastic gradient descent updates

3. **Advanced Training**:
   - Learning rate decay (reduces by 20% every 5 epochs)
   - Early stopping (stops if no improvement for 5 epochs)
   - Smaller batch sizes for better gradient estimates

4. **Visualization**:
   - Pixel overlay shows `image * weights` for the predicted class
   - Weight visualization shows learned patterns
   - Feature importance highlights most contributing pixels

The model learns to associate pixel patterns with digit classes through the weight matrix, which you can visualize to understand what the model "sees" when making predictions.

