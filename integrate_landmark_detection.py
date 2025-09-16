#!/usr/bin/env python3
"""
Integration script to add landmark detection to the existing image analyzer
"""

import os
import sys
import logging
from enhanced_landmark_detector import EnhancedLandmarkDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_landmark_detection():
    """Integrate landmark detection into enhanced_image_analyzer.py"""
    
    # Read the original enhanced_image_analyzer.py
    analyzer_file = "enhanced_image_analyzer.py"
    
    if not os.path.exists(analyzer_file):
        logger.error(f"❌ {analyzer_file} not found!")
        return False
    
    try:
        with open(analyzer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if landmark detection is already integrated
        if "EnhancedLandmarkDetector" in content:
            logger.info("✅ Landmark detection already integrated!")
            return True
        
        # Add import for landmark detector
        import_line = "from enhanced_landmark_detector import EnhancedLandmarkDetector"
        
        # Find the imports section and add our import
        lines = content.split('\n')
        import_section_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = i + 1
            elif line.strip() and not line.startswith('#') and not line.startswith('"""'):
                break
        
        # Insert the import
        lines.insert(import_section_end, import_line)
        
        # Find the __init__ method and add landmark detector initialization
        init_start = -1
        init_end = -1
        
        for i, line in enumerate(lines):
            if "def __init__(self" in line:
                init_start = i
            elif init_start != -1 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                init_end = i
                break
        
        if init_start != -1:
            # Find where to insert landmark detector initialization
            insert_pos = init_start + 1
            for i in range(init_start + 1, len(lines)):
                if "self.analysis_cache" in lines[i]:
                    insert_pos = i + 1
                    break
            
            # Add landmark detector initialization
            landmark_init = [
                "        # Initialize landmark detector",
                "        self.landmark_detector = EnhancedLandmarkDetector()",
                ""
            ]
            
            lines[insert_pos:insert_pos] = landmark_init
        
        # Find the analyze_image method and add landmark detection
        analyze_start = -1
        analyze_end = -1
        
        for i, line in enumerate(lines):
            if "def analyze_image(self" in line:
                analyze_start = i
            elif analyze_start != -1 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                analyze_end = i
                break
        
        if analyze_start != -1:
            # Find where to insert landmark detection
            insert_pos = analyze_start + 1
            for i in range(analyze_start + 1, len(lines)):
                if "analysis = {" in lines[i]:
                    insert_pos = i + 1
                    break
            
            # Add landmark detection to analysis
            landmark_analysis = [
                "                \"landmark_detection\": self._detect_landmarks(image_path),"
            ]
            
            lines[insert_pos:insert_pos] = landmark_analysis
        
        # Add the _detect_landmarks method
        method_to_add = [
            "",
            "    def _detect_landmarks(self, image_path):",
            "        \"\"\"Detect landmarks in the image\"\"\"",
            "        try:",
            "            return self.landmark_detector.detect_landmarks(image_path)",
            "        except Exception as e:",
            "            logger.error(f\"Error in landmark detection: {e}\")",
            "            return {\"error\": str(e)}",
            ""
        ]
        
        # Find the end of the class to add the method
        class_end = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "":
                continue
            if not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                class_end = i
                break
        
        if class_end != -1:
            lines[class_end:class_end] = method_to_add
        
        # Write the modified content back
        with open(analyzer_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info("✅ Successfully integrated landmark detection into enhanced_image_analyzer.py")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error integrating landmark detection: {e}")
        return False

def create_landmark_training_script():
    """Create a script to help with landmark training"""
    
    training_script = '''#!/usr/bin/env python3
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
        image_files = glob.glob(os.path.join(landmark_path, "*.jpg")) + \\
                     glob.glob(os.path.join(landmark_path, "*.jpeg")) + \\
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
'''
    
    with open("train_landmarks.py", 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    logger.info("✅ Created train_landmarks.py script")

def create_sample_data_structure():
    """Create sample data structure for training"""
    
    sample_structure = {
        "landmark_data": {
            "bitexco": {
                "description": "Bitexco Financial Tower images",
                "sample_files": ["bitexco_001.jpg", "bitexco_002.jpg", "bitexco_003.jpg"]
            },
            "landmark81": {
                "description": "Landmark 81 images", 
                "sample_files": ["landmark81_001.jpg", "landmark81_002.jpg", "landmark81_003.jpg"]
            },
            "ben_thanh": {
                "description": "Ben Thanh Market images",
                "sample_files": ["ben_thanh_001.jpg", "ben_thanh_002.jpg", "ben_thanh_003.jpg"]
            }
        }
    }
    
    # Create the directory structure
    os.makedirs("landmark_data", exist_ok=True)
    
    for landmark in sample_structure["landmark_data"]:
        landmark_dir = os.path.join("landmark_data", landmark)
        os.makedirs(landmark_dir, exist_ok=True)
        
        # Create a README file for each landmark
        readme_content = f"""# {landmark.upper()} Images

This folder should contain images of {sample_structure['landmark_data'][landmark]['description']}.

## Requirements:
- Image format: JPG, JPEG, or PNG
- Minimum size: 224x224 pixels
- Recommended size: 512x512 pixels or larger
- Minimum images: 50 per landmark
- Recommended images: 200+ per landmark

## Sample files:
{chr(10).join([f"- {file}" for file in sample_structure['landmark_data'][landmark]['sample_files']])}

## Tips:
- Include images from different angles
- Include images in different lighting conditions
- Include both day and night images if applicable
- Ensure images are clear and well-focused
"""
        
        with open(os.path.join(landmark_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    logger.info("✅ Created sample data structure in landmark_data/")

def main():
    """Main integration function"""
    
    print("🏛️ Landmark Detection Integration")
    print("=" * 40)
    
    # Step 1: Integrate into enhanced_image_analyzer.py
    print("\n1️⃣ Integrating landmark detection into enhanced_image_analyzer.py...")
    if integrate_landmark_detection():
        print("✅ Integration successful!")
    else:
        print("❌ Integration failed!")
        return
    
    # Step 2: Create training script
    print("\n2️⃣ Creating training script...")
    create_landmark_training_script()
    
    # Step 3: Create sample data structure
    print("\n3️⃣ Creating sample data structure...")
    create_sample_data_structure()
    
    print("\n🎉 Integration completed!")
    print("\n📋 Next steps:")
    print("1. Add your landmark images to the landmark_data/ folders")
    print("2. Run: python train_landmarks.py")
    print("3. The trained model will be automatically used by the image analyzer")
    print("\n📖 For more information, see the README files in landmark_data/ folders")

if __name__ == "__main__":
    main()

