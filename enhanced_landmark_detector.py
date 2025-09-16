#!/usr/bin/env python3
"""
Enhanced Landmark Detector for HCMC AI Challenge
"""

import os
import json
import logging
import numpy as np
import torch
from PIL import Image
import cv2
from datetime import datetime
import hashlib
from landmark_detection_trainer import LandmarkDetectionTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLandmarkDetector:
    def __init__(self, model_path="landmark_models/best_model.pth", cache_dir="landmark_cache"):
        self.model_path = model_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.trainer = LandmarkDetectionTrainer()
        self.model_loaded = False
        
        # Vietnamese landmark names
        self.vietnamese_landmarks = {
            "bitexco": {
                "vi": "Tòa nhà Bitexco Financial Tower",
                "en": "Bitexco Financial Tower",
                "description_vi": "Tòa nhà chọc trời biểu tượng của TP.HCM",
                "location": "District 1, Ho Chi Minh City"
            },
            "landmark81": {
                "vi": "Tòa nhà Landmark 81",
                "en": "Landmark 81",
                "description_vi": "Tòa nhà cao nhất Việt Nam",
                "location": "Vinhomes Central Park, District 1"
            },
            "ben_thanh": {
                "vi": "Chợ Bến Thành",
                "en": "Ben Thanh Market",
                "description_vi": "Chợ truyền thống nổi tiếng ở Quận 1",
                "location": "District 1, Ho Chi Minh City"
            }
        }
        
        self._load_model()
        self.analysis_cache = {}
        self._load_cache()
    
    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                success = self.trainer.load_model("best_model.pth")
                if success:
                    self.model_loaded = True
                    logger.info("✅ Landmark detection model loaded")
                else:
                    logger.warning("⚠️ Failed to load landmark model")
            else:
                logger.info("ℹ️ No trained model found")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    def _load_cache(self):
        cache_file = os.path.join(self.cache_dir, "landmark_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.analysis_cache = json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading cache: {e}")
    
    def _save_cache(self):
        cache_file = os.path.join(self.cache_dir, "landmark_cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving cache: {e}")
    
    def _get_image_hash(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return os.path.basename(image_path)
    
    def detect_landmarks(self, image_path, force_reanalyze=False):
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        image_hash = self._get_image_hash(image_path)
        if not force_reanalyze and image_hash in self.analysis_cache:
            return self.analysis_cache[image_hash]
        
        try:
            logger.info(f"🏛️ Detecting landmarks in: {os.path.basename(image_path)}")
            
            if not self.model_loaded:
                return self._basic_landmark_analysis(image_path)
            
            prediction = self.trainer.predict_landmark(image_path, confidence_threshold=0.3)
            
            if prediction:
                landmark_key = prediction['predicted_landmark']
                vietnamese_info = self.vietnamese_landmarks.get(landmark_key, {})
                
                result = {
                    "image_path": image_path,
                    "image_hash": image_hash,
                    "timestamp": datetime.now().isoformat(),
                    "landmark_detected": True,
                    "primary_landmark": {
                        "key": landmark_key,
                        "name_vi": vietnamese_info.get("vi", landmark_key),
                        "name_en": vietnamese_info.get("en", landmark_key),
                        "description_vi": vietnamese_info.get("description_vi", ""),
                        "location": vietnamese_info.get("location", ""),
                        "confidence": prediction['confidence']
                    },
                    "top_predictions": prediction['top_predictions']
                }
            else:
                result = {
                    "image_path": image_path,
                    "image_hash": image_hash,
                    "timestamp": datetime.now().isoformat(),
                    "landmark_detected": False
                }
            
            self.analysis_cache[image_hash] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error detecting landmarks: {e}")
            return {"error": str(e), "image_path": image_path}
    
    def _basic_landmark_analysis(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            height, width = image.shape[:2]
            
            analysis = {
                "image_path": image_path,
                "image_hash": self._get_image_hash(image_path),
                "timestamp": datetime.now().isoformat(),
                "landmark_detected": False,
                "analysis_type": "basic",
                "image_info": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 0
                },
                "suggestions": self._suggest_possible_landmarks(image)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in basic analysis: {e}")
            return {"error": str(e)}
    
    def _suggest_possible_landmarks(self, image):
        suggestions = []
        
        try:
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (height * width)
            
            if edge_density > 0.1:
                suggestions.append({
                    "landmark": "bitexco",
                    "confidence": 0.3,
                    "reason": "High edge density suggests modern buildings"
                })
            
            if aspect_ratio < 0.8:
                suggestions.append({
                    "landmark": "landmark81",
                    "confidence": 0.4,
                    "reason": "Tall aspect ratio suggests skyscraper"
                })
            
        except Exception as e:
            logger.error(f"Error suggesting landmarks: {e}")
        
        return suggestions
    
    def train_new_model(self, data_paths, labels):
        try:
            logger.info("🚀 Starting new model training...")
            
            train_loader, val_loader = self.trainer.prepare_data(data_paths, labels)
            self.trainer.create_model()
            best_acc = self.trainer.train(train_loader, val_loader)
            
            self._load_model()
            
            logger.info(f"✅ Model training completed: {best_acc:.2f}%")
            return best_acc
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return None

if __name__ == "__main__":
    detector = EnhancedLandmarkDetector()
    print("🏛️ Enhanced Landmark Detector initialized!")
    print(f"Model loaded: {detector.model_loaded}")
