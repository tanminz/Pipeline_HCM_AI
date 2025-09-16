#!/usr/bin/env python3
"""
Quick View CLIP Features
Xem nhanh các đặc trưng CLIP
"""

import numpy as np
import json
import pickle
import os

def quick_view_features():
    """Xem nhanh thông tin về CLIP features"""
    
    print("🔍 QUICK CLIP FEATURES VIEWER")
    print("="*50)
    
    # Check files
    files_to_check = [
        "fast_clip_features.npy",
        "fast_image_metadata.json", 
        "fast_valid_ids.pkl",
        "fast_faiss_index.bin"
    ]
    
    print("📁 Checking files:")
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file} (not found)")
    
    print("\n📊 Loading CLIP features...")
    
    # Load CLIP features
    if os.path.exists("fast_clip_features.npy"):
        features = np.load("fast_clip_features.npy")
        print(f"✅ Loaded features: {features.shape}")
        print(f"   - Number of images: {features.shape[0]}")
        print(f"   - Feature dimension: {features.shape[1]}")
        print(f"   - Data type: {features.dtype}")
        
        # Basic stats
        print(f"\n📈 Basic Statistics:")
        print(f"   - Mean: {np.mean(features):.6f}")
        print(f"   - Std:  {np.std(features):.6f}")
        print(f"   - Min:  {np.min(features):.6f}")
        print(f"   - Max:  {np.max(features):.6f}")
        
        # Check normalization
        norms = np.linalg.norm(features, axis=1)
        print(f"\n🎯 Normalization Check:")
        print(f"   - Mean norm: {np.mean(norms):.6f}")
        print(f"   - Std norm:  {np.std(norms):.6f}")
        
        if np.allclose(norms, 1.0, atol=1e-6):
            print("   ✅ Features are L2-normalized")
        else:
            print("   ⚠️ Features are NOT L2-normalized")
        
        # Show sample features
        print(f"\n🎯 Sample Features (first 3 images):")
        for i in range(min(3, features.shape[0])):
            feat = features[i]
            print(f"   Image {i+1}:")
            print(f"     - First 5 values: {feat[:5]}")
            print(f"     - Norm: {np.linalg.norm(feat):.6f}")
            print(f"     - Mean: {np.mean(feat):.6f}")
    
    # Load metadata
    if os.path.exists("fast_image_metadata.json"):
        with open("fast_image_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"\n📋 Metadata:")
        print(f"   - Total images: {len(metadata)}")
        
        # Show sample metadata
        sample_keys = list(metadata.keys())[:3]
        print(f"   - Sample image IDs: {sample_keys}")
        
        for key in sample_keys:
            meta = metadata[key]
            print(f"     {key}:")
            print(f"       - Path: {meta.get('web_path', 'N/A')}")
            print(f"       - Size: {meta.get('size', 'N/A')}")
    
    # Load valid IDs
    if os.path.exists("fast_valid_ids.pkl"):
        with open("fast_valid_ids.pkl", "rb") as f:
            valid_ids = pickle.load(f)
        print(f"\n🆔 Valid Image IDs:")
        print(f"   - Count: {len(valid_ids)}")
        print(f"   - Sample IDs: {valid_ids[:5]}")
    
    print("\n✅ Quick view complete!")

if __name__ == "__main__":
    quick_view_features()




