#!/usr/bin/env python3
"""
Landmark Detection Trainer for HCMC AI Challenge
Trains models to recognize famous landmarks like Bitexco, Landmark 81, etc.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import hashlib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandmarkDataset(Dataset):
    """Custom dataset for landmark detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LandmarkDetectionModel(nn.Module):
    """Custom model for landmark detection"""
    
    def __init__(self, num_classes, pretrained=True):
        super(LandmarkDetectionModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Modify final layer for our number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class LandmarkDetectionTrainer:
    """Trainer for landmark detection models"""
    
    def __init__(self, data_dir="landmark_data", model_dir="landmark_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Define landmarks to detect
        self.landmarks = {
            "bitexco": "Bitexco Financial Tower",
            "landmark81": "Landmark 81",
            "ben_thanh": "Ben Thanh Market",
            "notre_dame": "Notre Dame Cathedral",
            "reunification_palace": "Reunification Palace",
            "war_remnants": "War Remnants Museum",
            "cu_chi_tunnels": "Cu Chi Tunnels",
            "mekong_delta": "Mekong Delta",
            "phu_quoc": "Phu Quoc Island",
            "ha_long_bay": "Ha Long Bay",
            "hoan_kiem": "Hoan Kiem Lake",
            "temple_of_literature": "Temple of Literature",
            "other": "Other/Unknown"
        }
        
        self.label_to_idx = {landmark: idx for idx, landmark in enumerate(self.landmarks.keys())}
        self.idx_to_label = {idx: landmark for landmark, idx in self.label_to_idx.items()}
        
        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.image_size = (224, 224)
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model and training components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
    def prepare_data(self, image_paths, labels):
        """Prepare data for training"""
        logger.info("🔄 Preparing training data...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = LandmarkDataset(X_train, y_train, self.train_transform)
        val_dataset = LandmarkDataset(X_val, y_val, self.val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        logger.info(f"✅ Data prepared: {len(X_train)} training, {len(X_val)} validation samples")
        return train_loader, val_loader
    
    def create_model(self):
        """Create and initialize the model"""
        logger.info("🔄 Creating landmark detection model...")
        
        num_classes = len(self.landmarks)
        self.model = LandmarkDetectionModel(num_classes=num_classes, pretrained=True)
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        logger.info(f"✅ Model created with {num_classes} classes")
        return self.model
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(self.num_epochs):
            logger.info(f"📅 Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, predictions, labels = self.validate_epoch(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_model.pth")
                logger.info(f"💾 New best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save training history
        self.save_training_history(train_losses, train_accs, val_losses, val_accs)
        
        # Generate final evaluation report
        self.generate_evaluation_report(predictions, labels)
        
        logger.info("✅ Training completed!")
        return best_val_acc
    
    def save_model(self, filename):
        """Save model to disk"""
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'landmarks': self.landmarks,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'image_size': self.image_size
        }, model_path)
        logger.info(f"💾 Model saved to {model_path}")
    
    def load_model(self, filename):
        """Load model from disk"""
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model
        num_classes = len(checkpoint['landmarks'])
        self.model = LandmarkDetectionModel(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # Restore other components
        self.landmarks = checkpoint['landmarks']
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        self.image_size = checkpoint['image_size']
        
        logger.info(f"✅ Model loaded from {model_path}")
        return True
    
    def predict_landmark(self, image_path, confidence_threshold=0.5):
        """Predict landmark in a single image"""
        if self.model is None:
            logger.error("❌ Model not loaded. Please load a model first.")
            return None
        
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        predicted_label = self.idx_to_label[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        top_predictions = []
        
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.idx_to_label[idx.item()]
            top_predictions.append({
                'landmark': label,
                'name': self.landmarks[label],
                'confidence': prob.item()
            })
        
        result = {
            'predicted_landmark': predicted_label,
            'landmark_name': self.landmarks[predicted_label],
            'confidence': confidence_score,
            'top_predictions': top_predictions,
            'is_confident': confidence_score >= confidence_threshold
        }
        
        return result
    
    def save_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Save training history for plotting"""
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'epochs': list(range(1, len(train_losses) + 1))
        }
        
        history_path = os.path.join(self.model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create plots
        self.plot_training_history(history)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['epochs'], history['train_losses'], label='Train Loss')
        ax1.plot(history['epochs'], history['val_losses'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['epochs'], history['train_accs'], label='Train Accuracy')
        ax2.plot(history['epochs'], history['val_accs'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training history plots saved to {plot_path}")
    
    def generate_evaluation_report(self, predictions, labels):
        """Generate detailed evaluation report"""
        # Convert to class names
        pred_labels = [self.idx_to_label[pred] for pred in predictions]
        true_labels = [self.idx_to_label[label] for label in labels]
        
        # Classification report
        report = classification_report(true_labels, pred_labels, target_names=list(self.landmarks.keys()), output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=list(self.landmarks.keys()))
        
        # Save report
        report_path = os.path.join(self.model_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.landmarks.keys()),
                   yticklabels=list(self.landmarks.keys()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        cm_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📋 Evaluation report saved to {report_path}")
        logger.info(f"📊 Confusion matrix saved to {cm_path}")

def create_sample_data():
    """Create sample data structure for training"""
    sample_structure = {
        "bitexco": {
            "description": "Bitexco Financial Tower - Iconic skyscraper in Ho Chi Minh City",
            "sample_images": [
                "bitexco_day.jpg",
                "bitexco_night.jpg",
                "bitexco_aerial.jpg"
            ],
            "features": ["tall building", "helipad", "glass facade", "downtown"]
        },
        "landmark81": {
            "description": "Landmark 81 - Tallest building in Vietnam",
            "sample_images": [
                "landmark81_day.jpg",
                "landmark81_night.jpg",
                "landmark81_vue.jpg"
            ],
            "features": ["tallest building", "residential", "luxury", "vietnam"]
        },
        "ben_thanh": {
            "description": "Ben Thanh Market - Historic market in District 1",
            "sample_images": [
                "ben_thanh_exterior.jpg",
                "ben_thanh_interior.jpg",
                "ben_thanh_clock.jpg"
            ],
            "features": ["market", "historic", "clock tower", "shopping"]
        }
    }
    
    return sample_structure

if __name__ == "__main__":
    # Example usage
    trainer = LandmarkDetectionTrainer()
    
    # Create sample data structure
    sample_data = create_sample_data()
    print("Sample data structure created for landmarks:")
    for landmark, info in sample_data.items():
        print(f"- {landmark}: {info['description']}")
    
    print("\nTo train the model:")
    print("1. Organize your images in folders by landmark name")
    print("2. Use the prepare_data() method with your image paths and labels")
    print("3. Call train() to start training")
    print("4. Use predict_landmark() to make predictions on new images")

