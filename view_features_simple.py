#!/usr/bin/env python3
"""
Simple CLIP Features Viewer
Xem CLIP features từng phần để tránh memory issues
"""

import numpy as np
import json
import pickle
import os
import mmap

def view_features_simple():
    """Xem CLIP features một cách đơn giản"""
    
    print("🔍 SIMPLE CLIP FEATURES VIEWER")
    print("="*50)
    
    # Check file sizes
    files_to_check = [
        "fast_clip_features.npy",
        "fast_image_metadata.json", 
        "fast_valid_ids.pkl"
    ]
    
    print("📁 File Information:")
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file} (not found)")
    
    # Load metadata first
    print("\n📋 Loading metadata...")
    if os.path.exists("fast_image_metadata.json"):
        with open("fast_image_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata for {len(metadata)} images")
        
        # Show sample metadata
        sample_keys = list(metadata.keys())[:5]
        print(f"\n📸 Sample Images:")
        for i, key in enumerate(sample_keys):
            meta = metadata[key]
            print(f"   {i+1}. {key}")
            print(f"      Path: {meta.get('web_path', 'N/A')}")
            print(f"      Size: {meta.get('size', 'N/A')} bytes")
    
    # Load valid IDs
    print("\n🆔 Loading valid IDs...")
    if os.path.exists("fast_valid_ids.pkl"):
        with open("fast_valid_ids.pkl", "rb") as f:
            valid_ids = pickle.load(f)
        print(f"✅ Loaded {len(valid_ids)} valid image IDs")
        print(f"   Sample IDs: {valid_ids[:10]}")
    
    # Load features in chunks
    print("\n📊 Loading CLIP features in chunks...")
    if os.path.exists("fast_clip_features.npy"):
        # Get file info without loading entire array
        with open("fast_clip_features.npy", 'rb') as f:
            # Read header to get shape
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                print("❌ Not a valid NumPy file")
                return
            
            # Read version
            version = f.read(2)
            
            # Read header length
            header_len = int.from_bytes(f.read(2), byteorder='little')
            
            # Read header
            header = f.read(header_len).decode('ascii')
            
            # Parse shape from header
            import re
            shape_match = re.search(r"'shape':\s*\(([^)]+)\)", header)
            if shape_match:
                shape_str = shape_match.group(1)
                shape = tuple(int(x.strip()) for x in shape_str.split(','))
                print(f"✅ Features shape: {shape}")
                print(f"   - Number of images: {shape[0]}")
                print(f"   - Feature dimension: {shape[1]}")
                
                # Calculate total size
                total_elements = shape[0] * shape[1]
                total_size_mb = (total_elements * 4) / (1024*1024)  # 4 bytes per float32
                print(f"   - Total size: {total_size_mb:.1f} MB")
                
                # Load small sample
                print(f"\n🎯 Loading sample features (first 10 images)...")
                try:
                    # Load just the first 10 images
                    sample_features = np.load("fast_clip_features.npy", mmap_mode='r')[:10]
                    print(f"✅ Loaded sample: {sample_features.shape}")
                    
                    # Show statistics for sample
                    print(f"\n📈 Sample Statistics:")
                    print(f"   - Mean: {np.mean(sample_features):.6f}")
                    print(f"   - Std:  {np.std(sample_features):.6f}")
                    print(f"   - Min:  {np.min(sample_features):.6f}")
                    print(f"   - Max:  {np.max(sample_features):.6f}")
                    
                    # Check normalization
                    norms = np.linalg.norm(sample_features, axis=1)
                    print(f"\n🎯 Normalization Check:")
                    print(f"   - Mean norm: {np.mean(norms):.6f}")
                    print(f"   - Std norm:  {np.std(norms):.6f}")
                    
                    if np.allclose(norms, 1.0, atol=1e-6):
                        print("   ✅ Features are L2-normalized")
                    else:
                        print("   ⚠️ Features are NOT L2-normalized")
                    
                    # Show individual features
                    print(f"\n🔢 Individual Feature Vectors:")
                    for i in range(min(3, sample_features.shape[0])):
                        feat = sample_features[i]
                        print(f"   Image {i+1}:")
                        print(f"     - First 10 values: {feat[:10]}")
                        print(f"     - Norm: {np.linalg.norm(feat):.6f}")
                        print(f"     - Mean: {np.mean(feat):.6f}")
                        
                        # Show corresponding image ID
                        if i < len(valid_ids):
                            print(f"     - ID: {valid_ids[i]}")
                    
                except Exception as e:
                    print(f"❌ Error loading sample: {e}")
            else:
                print("❌ Could not parse shape from header")
    
    print("\n✅ Simple view complete!")

if __name__ == "__main__":
    view_features_simple()




