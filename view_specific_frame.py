#!/usr/bin/env python3
"""
View Specific Frame Features
Xem đặc trưng của một frame cụ thể
"""

import numpy as np
import json
import pickle
import os

def view_specific_frame(frame_id=None):
    """Xem đặc trưng của một frame cụ thể"""
    
    print("🎯 SPECIFIC FRAME FEATURE VIEWER")
    print("="*50)
    
    # Load data
    print("📂 Loading data...")
    
    # Load valid IDs
    with open("fast_valid_ids.pkl", "rb") as f:
        valid_ids = pickle.load(f)
    print(f"✅ Loaded {len(valid_ids)} valid image IDs")
    
    # Load metadata
    with open("fast_image_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata for {len(metadata)} images")
    
    # Load features with memory mapping
    features = np.load("fast_clip_features.npy", mmap_mode='r')
    print(f"✅ Loaded features: {features.shape}")
    
    # Select frame to view
    if frame_id is None:
        # Show available frames
        print(f"\n📋 Available frames (showing first 20):")
        for i, fid in enumerate(valid_ids[:20]):
            print(f"   {i+1:2d}. {fid}")
        
        # Select a sample frame
        frame_id = valid_ids[0]  # First frame
        print(f"\n🎯 Selected frame: {frame_id}")
    else:
        print(f"\n🎯 Viewing frame: {frame_id}")
    
    # Find frame index
    try:
        frame_index = valid_ids.index(frame_id)
        print(f"📊 Frame index: {frame_index}")
    except ValueError:
        print(f"❌ Frame ID not found: {frame_id}")
        return
    
    # Get features for this frame
    frame_features = features[frame_index]
    print(f"🎨 Feature vector shape: {frame_features.shape}")
    
    # Show feature statistics
    print(f"\n📈 Feature Statistics:")
    print(f"   - Mean: {np.mean(frame_features):.6f}")
    print(f"   - Std:  {np.std(frame_features):.6f}")
    print(f"   - Min:  {np.min(frame_features):.6f}")
    print(f"   - Max:  {np.max(frame_features):.6f}")
    print(f"   - Norm: {np.linalg.norm(frame_features):.6f}")
    
    # Show feature vector (first 20 values)
    print(f"\n🔢 Feature Vector (first 20 values):")
    for i, val in enumerate(frame_features[:20]):
        print(f"   [{i:2d}]: {val:8.6f}")
    print(f"   ... ({len(frame_features)-20} more values)")
    
    # Show metadata
    if frame_id in metadata:
        meta = metadata[frame_id]
        print(f"\n📋 Frame Metadata:")
        for key, value in meta.items():
            print(f"   - {key}: {value}")
    
    # Find similar frames
    print(f"\n🔍 Finding similar frames...")
    
    # Calculate similarities with other frames (sample)
    sample_size = min(1000, len(valid_ids))
    sample_indices = np.random.choice(len(valid_ids), sample_size, replace=False)
    sample_features = features[sample_indices]
    
    # Calculate similarities
    similarities = np.dot(sample_features, frame_features)
    
    # Find most similar frames (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:11]  # Top 10, excluding self
    
    print(f"📊 Most similar frames (from sample of {sample_size}):")
    for i, idx in enumerate(similar_indices):
        similar_frame_id = valid_ids[sample_indices[idx]]
        similarity = similarities[idx]
        print(f"   {i+1:2d}. {similar_frame_id} (similarity: {similarity:.6f})")
    
    print(f"\n✅ Frame analysis complete!")

def compare_two_frames(frame_id1, frame_id2):
    """So sánh hai frame"""
    
    print("🔄 FRAME COMPARISON")
    print("="*50)
    
    # Load data
    with open("fast_valid_ids.pkl", "rb") as f:
        valid_ids = pickle.load(f)
    
    features = np.load("fast_clip_features.npy", mmap_mode='r')
    
    # Find frame indices
    try:
        idx1 = valid_ids.index(frame_id1)
        idx2 = valid_ids.index(frame_id2)
    except ValueError as e:
        print(f"❌ Frame ID not found: {e}")
        return
    
    # Get features
    features1 = features[idx1]
    features2 = features[idx2]
    
    # Calculate similarity
    similarity = np.dot(features1, features2)
    distance = np.linalg.norm(features1 - features2)
    
    print(f"🎯 Frame 1: {frame_id1}")
    print(f"🎯 Frame 2: {frame_id2}")
    print(f"📊 Similarity (cosine): {similarity:.6f}")
    print(f"📏 Distance (L2): {distance:.6f}")
    
    # Show feature differences
    diff = features1 - features2
    print(f"\n📈 Feature Differences:")
    print(f"   - Mean difference: {np.mean(diff):.6f}")
    print(f"   - Std difference:  {np.std(diff):.6f}")
    print(f"   - Max difference:  {np.max(np.abs(diff)):.6f}")
    
    # Show largest differences
    largest_diff_indices = np.argsort(np.abs(diff))[-10:]
    print(f"\n🔍 Largest Feature Differences:")
    for idx in largest_diff_indices:
        print(f"   [{idx:3d}]: {diff[idx]:8.6f} (F1: {features1[idx]:8.6f}, F2: {features2[idx]:8.6f})")

if __name__ == "__main__":
    # View a specific frame
    view_specific_frame("L21_V001/152.jpg")
    
    print("\n" + "="*60)
    
    # Compare two frames
    compare_two_frames("L21_V001/152.jpg", "L21_V001/153.jpg")



