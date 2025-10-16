import numpy as np
import pickle
import os

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
    
    def train_model(self, X_train, y_train, num_epochs=5, batch_size=64):
        """Train the model on training data"""
        print("Training model...")
        
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
                
                loss = self.train_step(batch_X, batch_y)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        print("Model training completed!")
    
    def continue_training(self, X_train, y_train, additional_epochs=30):
        """Continue training the model with improved strategies"""
        print(f"Starting improved training with adaptive learning rate...")
        
        # Store original learning rate
        original_lr = self.learning_rate
        
        batch_size = 32  # Smaller batch size for better gradient estimates
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(additional_epochs):
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
                
                loss = self.train_step(batch_X, batch_y)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # Learning rate decay
            if epoch > 0 and epoch % 5 == 0:
                self.learning_rate *= 0.8
                print(f"Reduced learning rate to {self.learning_rate:.6f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{additional_epochs}, Average Loss: {avg_loss:.4f}, LR: {self.learning_rate:.6f}")
            
            # Early stopping
            if patience_counter >= patience and epoch > 10:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Restore original learning rate
        self.learning_rate = original_lr
        
        print(f"Improved training completed! Best loss: {best_loss:.4f}")
        return best_loss
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data"""
        print("\n" + "="*50)
        print("TESTING MODEL PERFORMANCE")
        print("="*50)
        
        # Make predictions
        predictions, probabilities = self.predict(X_test)
        
        # Calculate accuracy
        correct = np.sum(predictions == y_test)
        accuracy = correct / len(y_test)
        
        print(f"Test Accuracy: {accuracy:.3f} ({correct}/{len(y_test)})")
        
        # Show per-class accuracy
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        
        for i in range(len(y_test)):
            class_total[y_test[i]] += 1
            if predictions[i] == y_test[i]:
                class_correct[y_test[i]] += 1
        
        print("\nPer-class accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"  Class {i}: {acc:.3f} ({int(class_correct[i])}/{int(class_total[i])})")
        
        # Show average confidence
        avg_confidence = np.mean(np.max(probabilities, axis=1))
        print(f"\nAverage confidence: {avg_confidence:.3f}")
        
        print("="*50)
        
        return accuracy, avg_confidence
