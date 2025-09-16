#!/usr/bin/env python3
"""
View Video Frames
Xem các frame theo video
"""

import numpy as np
import json
import pickle
import os
from collections import defaultdict

def view_video_frames(video_name=None):
    """Xem các frame của một video cụ thể"""
    
    print("🎬 VIDEO FRAMES VIEWER")
    print("="*50)
    
    # Load data
    print("📂 Loading data...")
    
    with open("fast_valid_ids.pkl", "rb") as f:
        valid_ids = pickle.load(f)
    
    with open("fast_image_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    features = np.load("fast_clip_features.npy", mmap_mode='r')
    
    print(f"✅ Loaded {len(valid_ids)} frames")
    
    # Group frames by video
    video_frames = defaultdict(list)
    for frame_id in valid_ids:
        if '/' in frame_id:
            video = frame_id.split('/')[0]
            video_frames[video].append(frame_id)
    
    print(f"\n📹 Found {len(video_frames)} videos:")
    
    # Show video list
    video_list = sorted(video_frames.keys())
    for i, video in enumerate(video_list[:20]):  # Show first 20
        frame_count = len(video_frames[video])
        print(f"   {i+1:2d}. {video} ({frame_count} frames)")
    
    if len(video_list) > 20:
        print(f"   ... and {len(video_list) - 20} more videos")
    
    # Select video to view
    if video_name is None:
        video_name = video_list[0]  # First video
        print(f"\n🎯 Selected video: {video_name}")
    else:
        print(f"\n🎯 Viewing video: {video_name}")
    
    if video_name not in video_frames:
        print(f"❌ Video not found: {video_name}")
        return
    
    # Get frames for this video
    video_frame_ids = sorted(video_frames[video_name])
    print(f"📊 Video has {len(video_frame_ids)} frames")
    
    # Show first few frames
    print(f"\n📸 First 10 frames:")
    for i, frame_id in enumerate(video_frame_ids[:10]):
        frame_num = frame_id.split('/')[1].replace('.jpg', '')
        print(f"   {i+1:2d}. Frame {frame_num}: {frame_id}")
    
    # Analyze frame similarities within video
    print(f"\n🔍 Analyzing frame similarities within video...")
    
    # Get features for video frames
    video_frame_indices = [valid_ids.index(fid) for fid in video_frame_ids]
    video_frame_features = features[video_frame_indices]
    
    print(f"📊 Video frame features shape: {video_frame_features.shape}")
    
    # Calculate similarity matrix for first 20 frames
    sample_size = min(20, len(video_frame_features))
    sample_features = video_frame_features[:sample_size]
    similarity_matrix = np.dot(sample_features, sample_features.T)
    
    print(f"\n📈 Similarity analysis (first {sample_size} frames):")
    print(f"   - Mean similarity: {np.mean(similarity_matrix):.6f}")
    print(f"   - Std similarity:  {np.std(similarity_matrix):.6f}")
    print(f"   - Min similarity:  {np.min(similarity_matrix):.6f}")
    print(f"   - Max similarity:  {np.max(similarity_matrix):.6f}")
    
    # Find most similar consecutive frames
    consecutive_similarities = []
    for i in range(len(video_frame_features) - 1):
        sim = np.dot(video_frame_features[i], video_frame_features[i+1])
        consecutive_similarities.append(sim)
    
    if consecutive_similarities:
        print(f"\n🎬 Consecutive frame similarities:")
        print(f"   - Mean: {np.mean(consecutive_similarities):.6f}")
        print(f"   - Std:  {np.std(consecutive_similarities):.6f}")
        print(f"   - Min:  {np.min(consecutive_similarities):.6f}")
        print(f"   - Max:  {np.max(consecutive_similarities):.6f}")
        
        # Find most similar consecutive pair
        max_sim_idx = np.argmax(consecutive_similarities)
        max_sim = consecutive_similarities[max_sim_idx]
        frame1 = video_frame_ids[max_sim_idx]
        frame2 = video_frame_ids[max_sim_idx + 1]
        
        print(f"\n🎯 Most similar consecutive frames:")
        print(f"   - Frame 1: {frame1}")
        print(f"   - Frame 2: {frame2}")
        print(f"   - Similarity: {max_sim:.6f}")
        
        # Find least similar consecutive pair
        min_sim_idx = np.argmin(consecutive_similarities)
        min_sim = consecutive_similarities[min_sim_idx]
        frame1 = video_frame_ids[min_sim_idx]
        frame2 = video_frame_ids[min_sim_idx + 1]
        
        print(f"\n🎯 Least similar consecutive frames:")
        print(f"   - Frame 1: {frame1}")
        print(f"   - Frame 2: {frame2}")
        print(f"   - Similarity: {min_sim:.6f}")
    
    # Show feature evolution
    print(f"\n📊 Feature evolution analysis:")
    
    # Calculate feature statistics across frames
    feature_means = np.mean(video_frame_features, axis=0)
    feature_stds = np.std(video_frame_features, axis=0)
    
    print(f"   - Feature variation across frames:")
    print(f"     Mean of means: {np.mean(feature_means):.6f}")
    print(f"     Mean of stds:  {np.mean(feature_stds):.6f}")
    print(f"     Max std:       {np.max(feature_stds):.6f}")
    print(f"     Min std:       {np.min(feature_stds):.6f}")
    
    # Find most variable features
    most_variable_indices = np.argsort(feature_stds)[-10:]
    print(f"\n🔍 Most variable features across frames:")
    for i, idx in enumerate(most_variable_indices):
        std_val = feature_stds[idx]
        mean_val = feature_means[idx]
        print(f"   [{idx:3d}]: std={std_val:.6f}, mean={mean_val:.6f}")
    
    print(f"\n✅ Video analysis complete!")

def compare_videos(video1, video2):
    """So sánh hai video"""
    
    print("🔄 VIDEO COMPARISON")
    print("="*50)
    
    # Load data
    with open("fast_valid_ids.pkl", "rb") as f:
        valid_ids = pickle.load(f)
    
    features = np.load("fast_clip_features.npy", mmap_mode='r')
    
    # Group frames by video
    video_frames = defaultdict(list)
    for frame_id in valid_ids:
        if '/' in frame_id:
            video = frame_id.split('/')[0]
            video_frames[video].append(frame_id)
    
    if video1 not in video_frames or video2 not in video_frames:
        print(f"❌ One or both videos not found")
        return
    
    # Get frame features for both videos
    frames1 = sorted(video_frames[video1])
    frames2 = sorted(video_frames[video2])
    
    indices1 = [valid_ids.index(fid) for fid in frames1]
    indices2 = [valid_ids.index(fid) for fid in frames2]
    
    features1 = features[indices1]
    features2 = features[indices2]
    
    print(f"🎬 Video 1: {video1} ({len(frames1)} frames)")
    print(f"🎬 Video 2: {video2} ({len(frames2)} frames)")
    
    # Calculate average features for each video
    avg_features1 = np.mean(features1, axis=0)
    avg_features2 = np.mean(features2, axis=0)
    
    # Calculate similarity between average features
    similarity = np.dot(avg_features1, avg_features2)
    distance = np.linalg.norm(avg_features1 - avg_features2)
    
    print(f"\n📊 Video Similarity:")
    print(f"   - Similarity (cosine): {similarity:.6f}")
    print(f"   - Distance (L2): {distance:.6f}")
    
    # Calculate frame-to-frame similarities
    print(f"\n🔍 Frame-to-frame analysis:")
    
    # Sample frames for comparison
    sample_size = min(10, len(frames1), len(frames2))
    sample_features1 = features1[:sample_size]
    sample_features2 = features2[:sample_size]
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(sample_features1, sample_features2.T)
    
    print(f"   - Similarity matrix shape: {similarity_matrix.shape}")
    print(f"   - Mean similarity: {np.mean(similarity_matrix):.6f}")
    print(f"   - Max similarity: {np.max(similarity_matrix):.6f}")
    
    # Find most similar frame pair
    max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_idx]
    
    frame1 = frames1[max_idx[0]]
    frame2 = frames2[max_idx[1]]
    
    print(f"\n🎯 Most similar frame pair:")
    print(f"   - Frame 1: {frame1}")
    print(f"   - Frame 2: {frame2}")
    print(f"   - Similarity: {max_similarity:.6f}")

if __name__ == "__main__":
    # View frames of a specific video
    view_video_frames("L21_V001")
    
    print("\n" + "="*60)
    
    # Compare two videos
    compare_videos("L21_V001", "L21_V002")



