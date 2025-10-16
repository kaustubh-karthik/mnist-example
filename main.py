import tkinter as tk
import os
from model import SimpleNeuralNetwork
from data_loader import MNISTDataLoader
from ui_components import NNVisualizerUI

class NNVisualizerApp:
    """Main application class that coordinates model, data, and UI"""
    
    def __init__(self):
        self.root = tk.Tk()
        
        # Initialize components
        self.model = SimpleNeuralNetwork()
        self.data_loader = MNISTDataLoader()
        self.ui = NNVisualizerUI(self.root, self.model, self.data_loader)
        
        # Model file
        self.model_file = 'trained_model.pkl'
        
        # Set up callbacks
        self.ui.set_callbacks(
            improve_model_callback=self.improve_model,
            save_model_callback=self.save_model,
            image_changed_callback=self.on_image_changed
        )
        
        # Load or train model
        self.load_or_train_model()
        
        # Load initial image
        self.ui.load_current_image()
        self.ui.update_visualizations()
        
        # Clean up matplotlib figures when window is closed
        def on_closing():
            import matplotlib.pyplot as plt
            plt.close('all')
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        if os.path.exists(self.model_file):
            print("Loading existing model...")
            self.model.load_model(self.model_file)
        else:
            print("Training new model...")
            self.train_initial_model()
            self.model.save_model(self.model_file)
    
    def train_initial_model(self):
        """Train the model initially"""
        print("Loading MNIST data...")
        X_train, y_train = self.data_loader.load_training_data(max_samples=3000)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        self.model.train_model(X_train, y_train, num_epochs=5, batch_size=64)
    
    def improve_model(self):
        """Improve the model with advanced training"""
        print("Starting model improvement...")
        
        # Load more training data
        X_train, y_train = self.data_loader.load_training_data(max_samples=5000)
        
        # Continue training with improved strategies
        best_loss = self.model.continue_training(X_train, y_train, additional_epochs=30)
        
        # Save the improved model
        self.model.save_model(self.model_file)
        
        # Test the improved model
        self.test_model_performance()
        
        # Show success message
        self.ui.show_success_message("Model improved successfully!")
    
    def save_model(self):
        """Save the current model"""
        self.model.save_model(self.model_file)
        self.ui.show_success_message(f"Model saved to {self.model_file}")
    
    def on_image_changed(self):
        """Called when image changes in UI"""
        # This can be used for any additional processing when image changes
        pass
    
    def test_model_performance(self):
        """Test the model performance on test data"""
        print("Loading test data for performance evaluation...")
        X_test, y_test = self.data_loader.load_test_data(max_samples=100)
        
        # Evaluate model
        accuracy, avg_confidence = self.model.evaluate(X_test, y_test)
        
        print(f"\nModel Performance Summary:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function to start the application"""
    app = NNVisualizerApp()
    app.run()

if __name__ == "__main__":
    main()
