#!/usr/bin/env python3
"""
Landmark Training Helper Script
Helps you train the landmark detection model with your own data
"""

import os
import glob
from landmark_detection_trainer import LandmarkDetectionTrainer

def prepare_training_data(data_root="landmark_data"):
    """
    Prepare training data from organized folders
    
    Expected structure:
    landmark_data/
    ├── bitexco/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── landmark81/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ben_thanh/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    """
    
    image_paths = []
    labels = []
    
    if not os.path.exists(data_root):
        print(f"❌ Data directory {data_root} not found!")
        print("Please create the directory structure as shown above.")
        return None, None
    
    # Get all landmark folders
    landmark_folders = [d for d in os.listdir(data_root) 
                       if os.path.isdir(os.path.join(data_root, d))]
    
    print(f"📁 Found {len(landmark_folders)} landmark folders:")
    
    for landmark in landmark_folders:
        landmark_path = os.path.join(data_root, landmark)
        image_files = glob.glob(os.path.join(landmark_path, "*.jpg")) + \
                     glob.glob(os.path.join(landmark_path, "*.jpeg")) + \
                     glob.glob(os.path.join(landmark_path, "*.png"))
        
        print(f"  - {landmark}: {len(image_files)} images")
        
        # Add images and labels
        for image_file in image_files:
            image_paths.append(image_file)
            labels.append(landmark)
    
    print(f"📊 Total: {len(image_paths)} images")
    return image_paths, labels

def train_landmark_model():
    """Train the landmark detection model"""
    
    print("🚀 Starting landmark model training...")
    
    # Prepare data
    image_paths, labels = prepare_training_data()
    
    if not image_paths:
        print("❌ No training data found!")
        return
    
    # Create trainer and train
    trainer = LandmarkDetectionTrainer()
    
    print("📋 Data preparation...")
    train_loader, val_loader = trainer.prepare_data(image_paths, labels)
    
    print("🏗️ Creating model...")
    trainer.create_model()
    
    print("🎯 Starting training...")
    best_accuracy = trainer.train(train_loader, val_loader)
    
    print(f"✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
    print("💾 Model saved to landmark_models/best_model.pth")

if __name__ == "__main__":
    train_landmark_model()
