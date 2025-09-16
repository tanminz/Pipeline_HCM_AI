#!/usr/bin/env python3
"""
TransNet V2 Video Processor for AI Challenge V
Based on top 2 VBS 2020 team strategy
"""

import os
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SceneSegment:
    """Scene segment information"""
    start_frame: int
    end_frame: int
    confidence: float
    frames: List[int]  # Representative frames (first, middle, last)

class TransNetV2Processor:
    """TransNet V2 processor for scene change detection"""
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model = None
        self.model_path = model_path
        
        logger.info(f"TransNet V2 initialized on {self.device}")
    
    def load_model(self):
        """Load TransNet V2 model"""
        try:
            # Load TransNet V2 model
            # Note: In production, you would download the pre-trained model
            # For now, we'll use a simplified version
            logger.info("Loading TransNet V2 model...")
            
            # Simplified TransNet V2 architecture
            self.model = self._create_transnet_model()
            
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("TransNet V2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading TransNet V2 model: {e}")
            raise
    
    def _create_transnet_model(self):
        """Create simplified TransNet V2 model"""
        # Simplified version for demonstration
        # In production, use the actual TransNet V2 architecture
        model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model
    
    def extract_frames(self, video_path: str, fps: int = 1) -> List[np.ndarray]:
        """Extract frames from video at specified FPS"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        target_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(target_fps / fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def detect_scene_changes(self, frames: List[np.ndarray], 
                           threshold: float = 0.5) -> List[SceneSegment]:
        """Detect scene changes using TransNet V2"""
        if not self.model:
            self.load_model()
        
        segments = []
        current_segment_start = 0
        
        # Process frames in batches
        batch_size = 16
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare batch for model
            batch_tensor = self._prepare_batch(batch_frames)
            
            with torch.no_grad():
                predictions = self.model(batch_tensor)
                predictions = predictions.squeeze().cpu().numpy()
            
            # Detect scene changes
            for j, pred in enumerate(predictions):
                frame_idx = i + j
                
                if pred > threshold and frame_idx > current_segment_start:
                    # End current segment
                    segment = SceneSegment(
                        start_frame=current_segment_start,
                        end_frame=frame_idx - 1,
                        confidence=pred,
                        frames=self._get_representative_frames(
                            current_segment_start, frame_idx - 1
                        )
                    )
                    segments.append(segment)
                    current_segment_start = frame_idx
        
        # Add final segment
        if current_segment_start < len(frames):
            segment = SceneSegment(
                start_frame=current_segment_start,
                end_frame=len(frames) - 1,
                confidence=1.0,
                frames=self._get_representative_frames(
                    current_segment_start, len(frames) - 1
                )
            )
            segments.append(segment)
        
        logger.info(f"Detected {len(segments)} scene segments")
        return segments
    
    def _prepare_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Prepare batch of frames for TransNet V2"""
        # Resize frames to 224x224
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224))
            resized_frames.append(resized)
        
        # Convert to tensor and normalize
        batch = np.array(resized_frames)
        batch = batch.transpose(0, 3, 1, 2)  # (N, C, H, W)
        batch = batch / 255.0  # Normalize to [0, 1]
        
        # Add temporal dimension for 3D convolution
        batch = batch.reshape(1, batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])
        batch = torch.FloatTensor(batch).to(self.device)
        
        return batch
    
    def _get_representative_frames(self, start_frame: int, end_frame: int) -> List[int]:
        """Get representative frames: first, middle, last"""
        if start_frame == end_frame:
            return [start_frame]
        
        middle_frame = (start_frame + end_frame) // 2
        return [start_frame, middle_frame, end_frame]
    
    def filter_noise_segments(self, segments: List[SceneSegment], 
                            min_duration: int = 10) -> List[SceneSegment]:
        """Filter out noise segments (black screens, single color)"""
        filtered_segments = []
        
        for segment in segments:
            duration = segment.end_frame - segment.start_frame + 1
            
            # Filter very short segments (likely noise)
            if duration >= min_duration:
                filtered_segments.append(segment)
            else:
                logger.info(f"Filtered noise segment: {segment.start_frame}-{segment.end_frame}")
        
        logger.info(f"Filtered {len(segments) - len(filtered_segments)} noise segments")
        return filtered_segments

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, output_dir: str = "./processed_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.transnet = TransNetV2Processor()
        self.transnet.load_model()
        
        logger.info("Video processor initialized")
    
    def process_video(self, video_path: str, 
                     output_size: Tuple[int, int] = (200, 200),
                     quality: int = 85) -> Dict:
        """Process video and extract keyframes"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Extract frames at 1 FPS
        frames = self.transnet.extract_frames(str(video_path), fps=1)
        
        # Detect scene changes
        segments = self.transnet.detect_scene_changes(frames)
        
        # Filter noise
        segments = self.transnet.filter_noise_segments(segments)
        
        # Extract keyframes
        keyframes = self._extract_keyframes(frames, segments, output_size, quality)
        
        # Save metadata
        metadata = self._save_metadata(video_path, segments, keyframes)
        
        logger.info(f"Processed {len(keyframes)} keyframes from {video_path}")
        return metadata
    
    def _extract_keyframes(self, frames: List[np.ndarray], 
                          segments: List[SceneSegment],
                          output_size: Tuple[int, int],
                          quality: int) -> List[Dict]:
        """Extract and save keyframes"""
        keyframes = []
        
        for i, segment in enumerate(segments):
            for frame_idx in segment.frames:
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    # Resize frame
                    resized_frame = cv2.resize(frame, output_size)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                    
                    # Save keyframe
                    keyframe_path = self.output_dir / f"keyframe_{i:04d}_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(keyframe_path), frame_bgr, 
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
                    
                    keyframes.append({
                        'path': str(keyframe_path),
                        'segment_id': i,
                        'frame_id': frame_idx,
                        'start_frame': segment.start_frame,
                        'end_frame': segment.end_frame,
                        'confidence': segment.confidence
                    })
        
        return keyframes
    
    def _save_metadata(self, video_path: Path, segments: List[SceneSegment], 
                      keyframes: List[Dict]) -> Dict:
        """Save processing metadata"""
        metadata = {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'processing_time': time.time(),
            'total_segments': len(segments),
            'total_keyframes': len(keyframes),
            'segments': [
                {
                    'id': i,
                    'start_frame': seg.start_frame,
                    'end_frame': seg.end_frame,
                    'confidence': seg.confidence,
                    'representative_frames': seg.frames
                }
                for i, seg in enumerate(segments)
            ],
            'keyframes': keyframes
        }
        
        # Save metadata
        metadata_path = self.output_dir / f"{video_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def process_video_batch(self, video_dir: str, 
                          max_workers: int = 4) -> List[Dict]:
        """Process multiple videos in parallel"""
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_video, str(video)) 
                      for video in video_files]
            
            for future in futures:
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing video: {e}")
        
        return results

def main():
    """Main function for testing"""
    processor = VideoProcessor()
    
    # Example usage
    video_path = "path/to/your/video.mp4"
    
    if os.path.exists(video_path):
        try:
            metadata = processor.process_video(video_path)
            print(f"Processed video: {metadata['video_name']}")
            print(f"Extracted {metadata['total_keyframes']} keyframes")
            print(f"Found {metadata['total_segments']} scene segments")
        except Exception as e:
            logger.error(f"Error processing video: {e}")
    else:
        print(f"Video not found: {video_path}")

if __name__ == "__main__":
    main()


