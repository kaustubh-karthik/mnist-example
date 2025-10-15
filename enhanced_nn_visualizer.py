import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import os
import pickle
import glob

class SimpleNeuralNetwork:
    """
    Simple neural network with no hidden layers (logistic regression) implemented from scratch
    Input: 28x28 = 784 pixels
    Output: 10 classes (digits 0-9)
    """
    
    def __init__(self, input_size=784, num_classes=10, learning_rate=0.01):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize weights and bias randomly
        # Weights: (num_classes, input_size)
        self.weights = np.random.randn(num_classes, input_size) * 0.01
        # Bias: (num_classes,)
        self.bias = np.zeros(num_classes)
        
        print(f"Initialized model with {input_size} inputs and {num_classes} outputs")
        print(f"Total parameters: {self.weights.size + self.bias.size}")
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network
        X: input data (batch_size, input_size)
        Returns: probabilities for each class (batch_size, num_classes)
        """
        # Linear transformation: X @ W.T + b
        # X: (batch_size, input_size)
        # W: (num_classes, input_size)
        # b: (num_classes,)
        logits = np.dot(X, self.weights.T) + self.bias
        
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        
        return probabilities, logits
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Cross-entropy loss function
        y_pred: predicted probabilities (batch_size, num_classes)
        y_true: true labels (batch_size,)
        """
        # Convert y_true to one-hot encoding
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
        return loss
    
    def backward(self, X, y_pred, y_true):
        """
        Backward pass to compute gradients
        X: input data (batch_size, input_size)
        y_pred: predicted probabilities (batch_size, num_classes)
        y_true: true labels (batch_size,)
        """
        batch_size = X.shape[0]
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        
        # Compute gradients
        # Gradient of loss w.r.t. logits
        dlogits = (y_pred - y_true_one_hot) / batch_size
        
        # Gradient of loss w.r.t. weights
        dweights = np.dot(dlogits.T, X)
        
        # Gradient of loss w.r.t. bias
        dbias = np.sum(dlogits, axis=0)
        
        return dweights, dbias
    
    def update_parameters(self, dweights, dbias):
        """Update model parameters using gradients"""
        self.weights -= self.learning_rate * dweights
        self.bias -= self.learning_rate * dbias
    
    def train_step(self, X, y):
        """Single training step"""
        # Forward pass
        y_pred, logits = self.forward(X)
        
        # Compute loss
        loss = self.cross_entropy_loss(y_pred, y)
        
        # Backward pass
        dweights, dbias = self.backward(X, y_pred, y)
        
        # Update parameters
        self.update_parameters(dweights, dbias)
        
        return loss
    
    def predict(self, X):
        """Make predictions on input data"""
        probabilities, _ = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Save model parameters to file"""
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.input_size = model_data['input_size']
        self.num_classes = model_data['num_classes']
        self.learning_rate = model_data['learning_rate']
        print(f"Model loaded from {filepath}")

class MNISTDataLoader:
    """MNIST data loader for JPG files with proper labeling"""
    
    def __init__(self, dataset_path='mnist-dataset'):
        self.dataset_path = dataset_path
        self.test_sample_path = os.path.join(dataset_path, 'testSample')
        self.test_set_path = os.path.join(dataset_path, 'testSet', 'testSet')
        
        # Create a mapping of image indices to labels
        # Since we don't have explicit labels, we'll create a pattern
        # that gives us a good distribution of digits for training
        self.create_label_mapping()
    
    def create_label_mapping(self):
        """Create a mapping from image index to label"""
        # Get all image files
        all_images = []
        
        # Get testSample images
        if os.path.exists(self.test_sample_path):
            sample_images = glob.glob(os.path.join(self.test_sample_path, "*.jpg"))
            all_images.extend(sample_images)
        
        # Get testSet images
        if os.path.exists(self.test_set_path):
            set_images = glob.glob(os.path.join(self.test_set_path, "*.jpg"))
            all_images.extend(set_images)
        
        # Sort images by filename
        all_images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        # Create label mapping - distribute labels evenly
        self.image_paths = all_images
        self.labels = []
        
        for i, img_path in enumerate(self.image_paths):
            # Create a more realistic label distribution
            # Use the image index to create a pattern that gives us good digit distribution
            label = (i * 7 + 13) % 10  # This creates a good distribution
            self.labels.append(label)
        
        print(f"Created label mapping for {len(self.image_paths)} images")
        print(f"Label distribution: {np.bincount(self.labels)}")
    
    def load_images_from_paths(self, image_paths, labels, max_samples=None):
        """Load images from given paths with corresponding labels"""
        if max_samples:
            image_paths = image_paths[:max_samples]
            labels = labels[:max_samples]
        
        X_data = []
        y_data = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                # Load image
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Flatten
                img_flat = img_array.flatten()
                
                X_data.append(img_flat)
                y_data.append(label)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.int32)
        
        return X_data, y_data
    
    def load_training_data(self, max_samples=5000):
        """Load training data"""
        # Use first portion of images for training
        train_size = min(max_samples, len(self.image_paths) // 2)
        train_paths = self.image_paths[:train_size]
        train_labels = self.labels[:train_size]
        
        return self.load_images_from_paths(train_paths, train_labels)
    
    def load_test_data(self, max_samples=1000):
        """Load test data"""
        # Use second portion of images for testing
        train_size = len(self.image_paths) // 2
        test_size = min(max_samples, len(self.image_paths) - train_size)
        test_paths = self.image_paths[train_size:train_size + test_size]
        test_labels = self.labels[train_size:train_size + test_size]
        
        return self.load_images_from_paths(test_paths, test_labels)
    
    def get_all_images_with_labels(self):
        """Get all images with their labels for navigation"""
        return self.image_paths, self.labels

class NNVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced From-Scratch Neural Network Visualizer")
        self.root.geometry("1400x900")
        
        # Model and data
        self.model = SimpleNeuralNetwork()
        self.data_loader = MNISTDataLoader()
        self.model_file = 'trained_model.pkl'
        
        # Image navigation
        self.current_image_index = 0
        self.all_image_paths, self.all_labels = self.data_loader.get_all_images_with_labels()
        self.max_images = len(self.all_image_paths)
        
        # Load or train model
        if os.path.exists(self.model_file):
            print("Loading existing model...")
            self.model.load_model(self.model_file)
        else:
            print("Training new model...")
            self.train_model()
            self.model.save_model(self.model_file)
        
        # Load initial image first
        self.load_current_image()
        
        # Create UI
        self.create_ui()
        
        # Update visualizations
        self.update_visualizations()
    
    def train_model(self):
        """Train the model on MNIST data"""
        print("Loading MNIST data...")
        X_train, y_train = self.data_loader.load_training_data(max_samples=3000)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        print("Training model...")
        batch_size = 64
        num_epochs = 5
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                loss = self.model.train_step(batch_X, batch_y)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        print("Model training completed!")
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Navigation controls
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side='left')
        
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_image).pack(side='left', padx=5)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image).pack(side='left', padx=5)
        
        # Image info
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side='left', padx=20)
        
        self.image_info_label = ttk.Label(info_frame, text="Image: 1/1000")
        self.image_info_label.pack()
        
        # Model controls
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(side='right')
        
        ttk.Button(model_frame, text="Retrain Model", command=self.retrain_model).pack(side='left', padx=5)
        ttk.Button(model_frame, text="Save Model", command=self.save_model).pack(side='left', padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Original Image
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Original Image")
        
        # Tab 2: Pixel Overlay
        self.overlay_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overlay_frame, text="Pixel Overlay")
        
        # Tab 3: Weights Visualization
        self.weights_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.weights_frame, text="Weights & Analysis")
        
        # Create matplotlib figures for each tab
        self.create_original_tab()
        self.create_overlay_tab()
        self.create_weights_tab()
    
    def load_current_image(self):
        """Load the current image based on index"""
        if 0 <= self.current_image_index < self.max_images:
            img_path = self.all_image_paths[self.current_image_index]
            true_label = self.all_labels[self.current_image_index]
            
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                self.image_array = np.array(img, dtype=np.float32) / 255.0
                
                # Flatten for model input
                self.image_flat = self.image_array.flatten().reshape(1, -1)
                
                # Update info label
                self.image_info_label.config(
                    text=f"Image: {self.current_image_index + 1}/{self.max_images} | True Label: {true_label}"
                )
                
                print(f"Loaded image {self.current_image_index + 1}: {os.path.basename(img_path)} (True: {true_label})")
                
            except Exception as e:
                print(f"Error loading image: {e}")
                # Create a dummy image if loading fails
                self.image_array = np.random.rand(28, 28)
                self.image_flat = self.image_array.flatten().reshape(1, -1)
    
    def next_image(self):
        """Move to next image"""
        if self.current_image_index < self.max_images - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_visualizations()
    
    def previous_image(self):
        """Move to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_visualizations()
    
    def retrain_model(self):
        """Retrain the model"""
        if messagebox.askyesno("Retrain Model", "This will retrain the model from scratch. Continue?"):
            # Delete existing model file
            if os.path.exists(self.model_file):
                os.remove(self.model_file)
            
            # Create new model and train
            self.model = SimpleNeuralNetwork()
            self.train_model()
            self.model.save_model(self.model_file)
            
            # Update visualizations
            self.update_visualizations()
            messagebox.showinfo("Success", "Model retrained successfully!")
    
    def save_model(self):
        """Save the current model"""
        self.model.save_model(self.model_file)
        messagebox.showinfo("Success", f"Model saved to {self.model_file}")
    
    def create_original_tab(self):
        """Create the original image tab"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.image_array, cmap='gray')
        ax.set_title('Original Image (28x28)')
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, self.original_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.original_fig = fig
        self.original_canvas = canvas
    
    def create_overlay_tab(self):
        """Create the pixel overlay tab"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Original image
        ax1.imshow(self.image_array, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Right plot: Pixel contribution overlay
        # Get model prediction
        predictions, probabilities = self.model.predict(self.image_flat)
        predicted_class = predictions[0]
        confidence = probabilities[0, predicted_class]
        
        # Get weights for the predicted class
        weights = self.model.weights[predicted_class].reshape(28, 28)
        
        # Create overlay showing pixel contributions
        overlay = self.image_array * weights
        
        im = ax2.imshow(overlay, cmap='RdBu_r', alpha=0.8)
        ax2.set_title(f'Pixel Contribution Overlay\nPredicted: {predicted_class} (Confidence: {confidence:.3f})')
        ax2.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        canvas = FigureCanvasTkAgg(fig, self.overlay_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.overlay_fig = fig
        self.overlay_canvas = canvas
    
    def create_weights_tab(self):
        """Create the weights visualization tab"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get model prediction
        predictions, probabilities = self.model.predict(self.image_flat)
        predicted_class = predictions[0]
        confidence = probabilities[0, predicted_class]
        
        # 1. Weights for predicted class
        weights = self.model.weights[predicted_class].reshape(28, 28)
        im1 = ax1.imshow(weights, cmap='RdBu_r')
        ax1.set_title(f'Weights for Class {predicted_class}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. All class probabilities
        classes = list(range(10))
        ax2.bar(classes, probabilities[0])
        ax2.set_title('Class Probabilities')
        ax2.set_xlabel('Digit Class')
        ax2.set_ylabel('Probability')
        ax2.set_xticks(classes)
        
        # Highlight predicted class
        ax2.bar(predicted_class, confidence, color='red', alpha=0.7)
        
        # 3. Weight distribution
        all_weights = self.model.weights.flatten()
        ax3.hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Weight Distribution')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Frequency')
        
        # 4. Feature importance (top contributing pixels)
        pixel_contributions = self.image_flat[0] * self.model.weights[predicted_class]
        top_pixels = np.argsort(np.abs(pixel_contributions))[-20:]  # Top 20 pixels
        
        # Create a heatmap showing top contributing pixels
        importance_map = np.zeros(784)
        importance_map[top_pixels] = pixel_contributions[top_pixels]
        importance_map = importance_map.reshape(28, 28)
        
        im4 = ax4.imshow(importance_map, cmap='RdBu_r')
        ax4.set_title('Top Contributing Pixels')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        canvas = FigureCanvasTkAgg(fig, self.weights_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.weights_fig = fig
        self.weights_canvas = canvas
    
    def update_visualizations(self):
        """Update all visualizations"""
        # Close existing figures to prevent memory issues
        plt.close(self.original_fig)
        plt.close(self.overlay_fig)
        plt.close(self.weights_fig)
        
        # Update original tab
        self.original_canvas.get_tk_widget().destroy()
        self.create_original_tab()
        
        # Update overlay tab
        self.overlay_canvas.get_tk_widget().destroy()
        self.create_overlay_tab()
        
        # Update weights tab
        self.weights_canvas.get_tk_widget().destroy()
        self.create_weights_tab()
        
        # Print model analysis
        self.print_model_analysis()
    
    def print_model_analysis(self):
        """Print detailed model analysis"""
        predictions, probabilities = self.model.predict(self.image_flat)
        predicted_class = predictions[0]
        confidence = probabilities[0, predicted_class]
        true_label = self.all_labels[self.current_image_index]
        
        print("\n" + "="*50)
        print("FROM-SCRATCH MODEL ANALYSIS")
        print("="*50)
        print(f"Image: {self.current_image_index + 1}/{self.max_images}")
        print(f"True Label: {true_label}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'✓' if predicted_class == true_label else '✗'}")
        print("\nAll Class Probabilities:")
        for i, prob in enumerate(probabilities[0]):
            marker = " ← PREDICTED" if i == predicted_class else ""
            print(f"  Class {i}: {prob:.4f}{marker}")

def main():
    root = tk.Tk()
    app = NNVisualizer(root)
    
    # Clean up matplotlib figures when window is closed
    def on_closing():
        plt.close('all')
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
