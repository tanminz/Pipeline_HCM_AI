#!/usr/bin/env python3
"""
Test script for landmark detection system
"""

import os
import sys
import logging
from enhanced_landmark_detector import EnhancedLandmarkDetector
from landmark_detection_trainer import LandmarkDetectionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_landmark_detector():
    """Test the landmark detection system"""
    
    print("🏛️ Testing Landmark Detection System")
    print("=" * 40)
    
    # Initialize detector
    detector = EnhancedLandmarkDetector()
    
    print(f"✅ Detector initialized")
    print(f"📊 Model loaded: {detector.model_loaded}")
    
    # Test with sample images if available
    test_images = []
    
    # Look for test images in static/images directory
    static_images_dir = "static/images"
    if os.path.exists(static_images_dir):
        image_extensions = ['.jpg', '.jpeg', '.png']
        for ext in image_extensions:
            test_images.extend(glob.glob(os.path.join(static_images_dir, f"*{ext}")))
    
    # Also check current directory
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend(glob.glob(f"*{ext}"))
    
    if test_images:
        print(f"\n📸 Found {len(test_images)} test images")
        
        # Test first few images
        for i, image_path in enumerate(test_images[:3]):
            print(f"\n🔍 Testing image {i+1}: {os.path.basename(image_path)}")
            
            try:
                result = detector.detect_landmarks(image_path)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    if result.get("landmark_detected", False):
                        landmark = result["primary_landmark"]
                        print(f"✅ Detected: {landmark['name_vi']} ({landmark['name_en']})")
                        print(f"📍 Location: {landmark['location']}")
                        print(f"🎯 Confidence: {landmark['confidence']:.2%}")
                        
                        if "top_predictions" in result:
                            print("🏆 Top predictions:")
                            for j, pred in enumerate(result["top_predictions"][:3]):
                                print(f"   {j+1}. {pred['name']}: {pred['confidence']:.2%}")
                    else:
                        print("❓ No landmark detected")
                        
                        if "suggestions" in result:
                            print("💡 Suggestions:")
                            for suggestion in result["suggestions"]:
                                print(f"   - {suggestion['landmark']}: {suggestion['confidence']:.2%} ({suggestion['reason']})")
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
    else:
        print("\n📸 No test images found")
        print("💡 To test the system:")
        print("   1. Add some images to the current directory")
        print("   2. Or add images to static/images/ directory")
        print("   3. Run this script again")

def test_training_system():
    """Test the training system"""
    
    print("\n🎯 Testing Training System")
    print("=" * 30)
    
    # Initialize trainer
    trainer = LandmarkDetectionTrainer()
    
    print(f"✅ Trainer initialized")
    print(f"📊 Available landmarks: {len(trainer.landmarks)}")
    
    # Show available landmarks
    print("\n🏛️ Available landmarks:")
    for key, name in trainer.landmarks.items():
        print(f"   - {key}: {name}")
    
    # Check if training data exists
    data_dir = "landmark_data"
    if os.path.exists(data_dir):
        print(f"\n📁 Training data directory found: {data_dir}")
        
        landmark_folders = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        
        if landmark_folders:
            print(f"📊 Found {len(landmark_folders)} landmark folders:")
            total_images = 0
            
            for landmark in landmark_folders:
                landmark_path = os.path.join(data_dir, landmark)
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend(glob.glob(os.path.join(landmark_path, f"*{ext}")))
                
                print(f"   - {landmark}: {len(image_files)} images")
                total_images += len(image_files)
            
            print(f"📈 Total images: {total_images}")
            
            if total_images >= 50:
                print("✅ Sufficient data for training!")
                print("💡 Run: python train_landmarks.py to start training")
            else:
                print("⚠️ Need more training data (minimum 50 images recommended)")
        else:
            print("❌ No landmark folders found")
            print("💡 Create folders for each landmark and add images")
    else:
        print(f"❌ Training data directory not found: {data_dir}")
        print("💡 Run: python integrate_landmark_detection.py to create the structure")

def create_demo_data():
    """Create demo data for testing"""
    
    print("\n🎬 Creating Demo Data")
    print("=" * 25)
    
    # Create demo directory
    demo_dir = "demo_landmarks"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create sample landmark folders
    landmarks = ["bitexco", "landmark81", "ben_thanh"]
    
    for landmark in landmarks:
        landmark_dir = os.path.join(demo_dir, landmark)
        os.makedirs(landmark_dir, exist_ok=True)
        
        # Create a sample text file explaining what should be in this folder
        readme_content = f"""# {landmark.upper()} Demo Folder

This folder should contain images of {landmark}.

## Example images to add:
- {landmark}_day.jpg
- {landmark}_night.jpg  
- {landmark}_aerial.jpg
- {landmark}_closeup.jpg

## Image requirements:
- Format: JPG, JPEG, or PNG
- Size: At least 224x224 pixels
- Quality: Clear, well-focused images

## Tips:
- Include images from different angles
- Include images in different lighting conditions
- The more diverse the images, the better the model will perform
"""
        
        with open(os.path.join(landmark_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    print(f"✅ Created demo structure in {demo_dir}/")
    print("💡 Add your landmark images to the respective folders")
    print("💡 Then run: python train_landmarks.py")

def show_usage_examples():
    """Show usage examples"""
    
    print("\n📖 Usage Examples")
    print("=" * 20)
    
    examples = """
# 1. Basic landmark detection
from enhanced_landmark_detector import EnhancedLandmarkDetector

detector = EnhancedLandmarkDetector()
result = detector.detect_landmarks("path/to/image.jpg")

if result.get("landmark_detected"):
    landmark = result["primary_landmark"]
    print(f"Detected: {landmark['name_vi']}")
    print(f"Confidence: {landmark['confidence']:.2%}")

# 2. Training a new model
from landmark_detection_trainer import LandmarkDetectionTrainer

trainer = LandmarkDetectionTrainer()
train_loader, val_loader = trainer.prepare_data(image_paths, labels)
trainer.create_model()
best_acc = trainer.train(train_loader, val_loader)

# 3. Integration with image analyzer
from enhanced_image_analyzer import EnhancedImageAnalyzer

analyzer = EnhancedImageAnalyzer()
analysis = analyzer.analyze_image("path/to/image.jpg")

# Landmark detection results will be in:
landmark_result = analysis.get("landmark_detection", {})
"""
    
    print(examples)

def main():
    """Main test function"""
    
    import glob
    
    print("🏛️ Landmark Detection System Test")
    print("=" * 40)
    
    # Test 1: Landmark detector
    test_landmark_detector()
    
    # Test 2: Training system
    test_training_system()
    
    # Test 3: Create demo data
    create_demo_data()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Testing completed!")
    print("\n📋 Next steps:")
    print("1. Add landmark images to demo_landmarks/ folders")
    print("2. Run: python train_landmarks.py")
    print("3. Test detection with: python test_landmark_detection.py")

if __name__ == "__main__":
    main()

