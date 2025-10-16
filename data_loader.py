import numpy as np
import os
import glob
import pickle
from PIL import Image

class MNISTDataLoader:
    """MNIST data loader for JPG files with proper labeling"""
    
    def __init__(self, dataset_path='mnist-dataset'):
        self.dataset_path = dataset_path
        self.digit_folders = [str(i) for i in range(10)]  # 0-9
        self.testset_path = 'testSet'
        
        # Image cache for instant loading
        self.image_cache = {}  # {image_path: (image_array, image_flat)}
        
        # TestSet prediction cache
        self.testset_cache_file = 'testset_predictions.pkl'
        self.testset_predictions = {}  # {image_path: (prediction, probabilities)}
        
        # Current dataset mode ('training' or 'testset')
        self.current_dataset = 'training'
        
        # Create a mapping of image paths to labels using folder structure
        self.create_label_mapping()
        
        # Load testSet predictions cache if it exists
        self.load_testset_cache()
        
        # Pre-load a batch of images for instant navigation
        self.preload_images()
    
    def create_label_mapping(self):
        """Create a mapping from image paths to labels using folder structure"""
        # Start with training images
        self.image_paths, self.labels = self.get_training_images_with_labels()
        
        print(f"Created label mapping for {len(self.image_paths)} training images")
        if len(self.labels) > 0:
            print(f"Label distribution: {np.bincount(self.labels)}")
            print(f"Available digits: {sorted(set(self.labels))}")
            print("Images shuffled for random navigation order")
        else:
            print("No training images found in the dataset!")
    
    def preload_images(self, num_images=500):
        """Pre-load images for instant navigation"""
        print(f"Pre-loading {min(num_images, len(self.image_paths))} images for instant navigation...")
        
        for i, img_path in enumerate(self.image_paths[:num_images]):
            try:
                # Load and process image
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Flatten for model input
                img_flat = img_array.flatten().reshape(1, -1)
                
                # Cache the processed image
                self.image_cache[img_path] = (img_array, img_flat)
                
                # Progress indicator (less frequent for faster startup)
                if (i + 1) % 200 == 0 or (i + 1) == min(num_images, len(self.image_paths)):
                    print(f"Pre-loaded {i + 1}/{min(num_images, len(self.image_paths))} images...")
                    
            except Exception as e:
                print(f"Error pre-loading {img_path}: {e}")
                continue
        
        print(f"Pre-loading complete! {len(self.image_cache)} images cached.")
    
    def load_testset_cache(self):
        """Load testSet predictions cache from file"""
        if os.path.exists(self.testset_cache_file):
            try:
                with open(self.testset_cache_file, 'rb') as f:
                    self.testset_predictions = pickle.load(f)
                print(f"Loaded testSet predictions cache: {len(self.testset_predictions)} images")
            except Exception as e:
                print(f"Error loading testSet cache: {e}")
                self.testset_predictions = {}
        else:
            print("No testSet predictions cache found")
    
    def save_testset_cache(self):
        """Save testSet predictions cache to file"""
        try:
            with open(self.testset_cache_file, 'wb') as f:
                pickle.dump(self.testset_predictions, f)
            print(f"Saved testSet predictions cache: {len(self.testset_predictions)} images")
        except Exception as e:
            print(f"Error saving testSet cache: {e}")
    
    def get_testset_images(self):
        """Get all testSet image paths"""
        if not os.path.exists(self.testset_path):
            return []
        
        # Get all JPG files in testSet
        testset_images = glob.glob(os.path.join(self.testset_path, "*.jpg"))
        testset_images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        return testset_images
    
    def cache_testset_predictions(self, model):
        """Run model predictions on all testSet images and cache results"""
        testset_images = self.get_testset_images()
        
        if not testset_images:
            print("No testSet images found!")
            return
        
        print(f"Running model predictions on {len(testset_images)} testSet images...")
        
        for i, img_path in enumerate(testset_images):
            try:
                # Load and process image
                img = Image.open(img_path).convert('L')
                
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_flat = img_array.flatten().reshape(1, -1)
                
                # Get model prediction
                predictions, probabilities = model.predict(img_flat)
                prediction = predictions[0]
                
                # Cache the prediction
                self.testset_predictions[img_path] = (prediction, probabilities[0])
                
                # Progress indicator
                if (i + 1) % 1000 == 0 or (i + 1) == len(testset_images):
                    print(f"Processed {i + 1}/{len(testset_images)} testSet images...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Save cache to file
        self.save_testset_cache()
        print(f"TestSet predictions caching complete! {len(self.testset_predictions)} images cached.")
    
    def switch_dataset(self, dataset_name):
        """Switch between training and testSet datasets"""
        if dataset_name == 'training':
            self.current_dataset = 'training'
            self.image_paths, self.labels = self.get_training_images_with_labels()
        elif dataset_name == 'testset':
            self.current_dataset = 'testset'
            self.image_paths, self.labels = self.get_testset_images_with_labels()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.max_images = len(self.image_paths)
        print(f"Switched to {dataset_name} dataset: {self.max_images} images")
    
    def get_training_images_with_labels(self):
        """Get training images with labels"""
        image_paths = []
        labels = []
        
        # Load images from each digit folder
        for digit_folder in self.digit_folders:
            digit_path = os.path.join(self.dataset_path, digit_folder)
            
            if os.path.exists(digit_path):
                # Get all JPG files in this digit folder
                jpg_files = glob.glob(os.path.join(digit_path, "*.jpg"))
                
                # Add images and their corresponding labels
                for img_path in jpg_files:
                    image_paths.append(img_path)
                    labels.append(int(digit_folder))  # Label is the folder name
        
        # Shuffle the images and labels together to get random order
        combined = list(zip(image_paths, labels))
        np.random.shuffle(combined)
        image_paths, labels = zip(*combined)
        
        # Convert back to lists
        return list(image_paths), list(labels)
    
    def get_testset_images_with_labels(self):
        """Get testSet images with dummy labels (since they're unlabeled)"""
        testset_images = self.get_testset_images()
        # Use dummy labels since testSet images are unlabeled
        dummy_labels = [0] * len(testset_images)  # All labeled as 0 for display purposes
        return testset_images, dummy_labels
    
    def get_cached_prediction(self, image_path):
        """Get cached prediction for a testSet image"""
        if image_path in self.testset_predictions:
            return self.testset_predictions[image_path]
        return None, None
    
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
        """Load training data with balanced sampling from each class"""
        # Get balanced samples from each digit class
        train_paths = []
        train_labels = []
        
        # Group images by label
        images_by_label = {}
        for img_path, label in zip(self.image_paths, self.labels):
            if label not in images_by_label:
                images_by_label[label] = []
            images_by_label[label].append(img_path)
        
        # Sample from each class
        samples_per_class = max_samples // len(images_by_label) if images_by_label else 0
        
        for label, img_paths in images_by_label.items():
            # Take up to samples_per_class images from this class
            class_samples = min(samples_per_class, len(img_paths))
            selected_paths = img_paths[:class_samples]
            
            train_paths.extend(selected_paths)
            train_labels.extend([label] * class_samples)
        
        print(f"Loading {len(train_paths)} training samples")
        return self.load_images_from_paths(train_paths, train_labels)
    
    def load_test_data(self, max_samples=1000):
        """Load test data with balanced sampling from each class"""
        # Get balanced samples from each digit class for testing
        test_paths = []
        test_labels = []
        
        # Group images by label
        images_by_label = {}
        for img_path, label in zip(self.image_paths, self.labels):
            if label not in images_by_label:
                images_by_label[label] = []
            images_by_label[label].append(img_path)
        
        # Sample from each class for testing
        samples_per_class = max_samples // len(images_by_label) if images_by_label else 0
        
        for label, img_paths in images_by_label.items():
            # Take samples from the end of each class (different from training)
            class_samples = min(samples_per_class, len(img_paths))
            start_idx = max(0, len(img_paths) - class_samples)
            selected_paths = img_paths[start_idx:]
            
            test_paths.extend(selected_paths)
            test_labels.extend([label] * len(selected_paths))
        
        print(f"Loading {len(test_paths)} test samples")
        return self.load_images_from_paths(test_paths, test_labels)
    
    def get_all_images_with_labels(self):
        """Get all images with their labels for navigation"""
        return self.image_paths, self.labels
    
    def load_single_image(self, image_path):
        """Load a single image for visualization (uses cache for instant loading)"""
        # Check if image is in cache
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        # If not in cache, load it (this should be rare after preloading)
        try:
            print(f"Loading uncached image: {os.path.basename(image_path)}")
            
            # Load image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to 28x28 if needed
            if img.size != (28, 28):
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Flatten for model input
            img_flat = img_array.flatten().reshape(1, -1)
            
            # Cache it for future use
            self.image_cache[image_path] = (img_array, img_flat)
            
            return img_array, img_flat
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            img_array = np.random.rand(28, 28)
            img_flat = img_array.flatten().reshape(1, -1)
            return img_array, img_flat
