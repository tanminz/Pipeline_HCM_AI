#!/usr/bin/env python3
"""
Install Enhanced Dependencies for HCMC AI Challenge
Installs OCR and image analysis libraries
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all enhanced dependencies"""
    print("🚀 Installing Enhanced Dependencies for HCMC AI Challenge...")
    
    # Core OCR and image analysis packages
    packages = [
        "easyocr>=1.7.0",
        "PaddleOCR>=2.7.0", 
        "pytesseract>=0.3.10",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0"
    ]
    
    # Additional utilities
    additional_packages = [
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    all_packages = packages + additional_packages
    
    success_count = 0
    total_count = len(all_packages)
    
    for package in all_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_count} packages")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        print("\n🔧 Next steps:")
        print("1. Run: python app.py")
        print("2. Access the enhanced search at: http://localhost:5000")
        print("3. Try searching with text, objects, or scene descriptions")
    else:
        print("⚠️ Some packages failed to install. Check the errors above.")
        print("You can try installing them manually or check your Python environment.")

if __name__ == "__main__":
    main()






