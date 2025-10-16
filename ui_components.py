import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import os

class NNVisualizerUI:
    """UI components for the Neural Network Visualizer"""
    
    def __init__(self, root, model, data_loader):
        self.root = root
        self.root.title("Enhanced From-Scratch Neural Network Visualizer")
        self.root.geometry("1400x900")
        
        self.model = model
        self.data_loader = data_loader
        
        # Image navigation
        self.current_image_index = 0
        self.all_image_paths, self.all_labels = self.data_loader.get_all_images_with_labels()
        self.max_images = len(self.all_image_paths)
        
        # Current image data
        self.image_array = None
        self.image_flat = None
        
        # Callbacks for external functions
        self.on_improve_model = None
        self.on_save_model = None
        self.on_image_changed = None
        
        self.create_ui()
    
    def set_callbacks(self, improve_model_callback, save_model_callback, image_changed_callback):
        """Set callback functions for external actions"""
        self.on_improve_model = improve_model_callback
        self.on_save_model = save_model_callback
        self.on_image_changed = image_changed_callback
    
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
        
        ttk.Button(model_frame, text="Improve Model", command=self.retrain_model).pack(side='left', padx=5)
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
        
        # Initialize with dummy data
        self.image_array = np.random.rand(28, 28)
        self.image_flat = self.image_array.flatten().reshape(1, -1)
        
        # Create matplotlib figures for each tab
        self.create_original_tab()
        self.create_overlay_tab()
        self.create_weights_tab()
    
    def load_current_image(self):
        """Load the current image based on index"""
        if 0 <= self.current_image_index < self.max_images:
            img_path = self.all_image_paths[self.current_image_index]
            true_label = self.all_labels[self.current_image_index]
            
            self.image_array, self.image_flat = self.data_loader.load_single_image(img_path)
            
            # Update info label
            self.image_info_label.config(
                text=f"Image: {self.current_image_index + 1}/{self.max_images} | True Label: {true_label}"
            )
            
            print(f"Loaded image {self.current_image_index + 1}: {os.path.basename(img_path)} (True: {true_label})")
            
            # Notify external callback
            if self.on_image_changed:
                self.on_image_changed()
    
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
        """Trigger model improvement"""
        if messagebox.askyesno("Continue Training", "This will add 20 more training epochs to the existing model. Continue?"):
            if self.on_improve_model:
                self.on_improve_model()
            self.update_visualizations()
    
    def save_model(self):
        """Trigger model save"""
        if self.on_save_model:
            self.on_save_model()
    
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
    
    def show_success_message(self, message):
        """Show success message"""
        messagebox.showinfo("Success", message)
