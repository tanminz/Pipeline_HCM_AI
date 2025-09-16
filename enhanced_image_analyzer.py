from enhanced_landmark_detector import EnhancedLandmarkDetector
#!/usr/bin/env python3
"""
Enhanced Image Analyzer for HCMC AI Challenge
Provides OCR, object detection, and metadata extraction capabilities
"""

import os
import json
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    DetrImageProcessor, 
    DetrForObjectDetection,
    AutoImageProcessor,
    AutoModelForImageClassification
)
import easyocr
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImageAnalyzer:
    """Advanced image analysis with OCR, object detection, and metadata extraction"""
    
    def __init__(self, cache_dir="image_analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models
        self.ocr_reader = None
        self.ocr_processor = None
        self.ocr_model = None
        self.object_detector = None
        self.object_processor = None
        self.image_classifier = None
        self.classifier_processor = None
        
        # Load models
        self._load_models()
        
        # Analysis cache
        self.analysis_cache = {}
        # Initialize landmark detector
        self.landmark_detector = EnhancedLandmarkDetector()

        self._load_cache()
    
    def _load_models(self):
        """Load AI models for analysis"""
        try:
            logger.info("🔄 Loading OCR models...")
            
            # Load EasyOCR for Vietnamese and English text
            self.ocr_reader = easyocr.Reader(['en', 'vi'], gpu=torch.cuda.is_available())
            logger.info("✅ EasyOCR loaded successfully")
            
            # Load TrOCR for better text recognition
            try:
                self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                logger.info("✅ TrOCR loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ TrOCR loading failed: {e}")
            
            # Load object detection model
            try:
                logger.info("🔄 Loading object detection model...")
                self.object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                logger.info("✅ Object detection model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Object detection loading failed: {e}")
            
            # Load image classification model
            try:
                logger.info("🔄 Loading image classification model...")
                self.classifier_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
                self.image_classifier = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
                logger.info("✅ Image classification model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Image classification loading failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def _load_cache(self):
        """Load analysis cache from disk"""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.analysis_cache = json.load(f)
                logger.info(f"✅ Loaded {len(self.analysis_cache)} cached analyses")
            except Exception as e:
                logger.error(f"❌ Error loading cache: {e}")
    
    def _save_cache(self):
        """Save analysis cache to disk"""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving cache: {e}")
    
    def _get_image_hash(self, image_path):
        """Generate hash for image to use as cache key"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return os.path.basename(image_path)
    
    def analyze_image(self, image_path, force_reanalyze=False):
        """Comprehensive image analysis including OCR, objects, and metadata"""
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        # Check cache first
        image_hash = self._get_image_hash(image_path)
        if not force_reanalyze and image_hash in self.analysis_cache:
            logger.info(f"📋 Using cached analysis for {os.path.basename(image_path)}")
            return self.analysis_cache[image_hash]
        
        try:
            logger.info(f"🔍 Analyzing image: {os.path.basename(image_path)}")
            
            # Load image
            image = Image.open(image_path)
            
            # Perform all analyses
            analysis = {
                "landmark_detection": self._detect_landmarks(image_path),
                "image_path": image_path,
                "image_hash": image_hash,
                "timestamp": datetime.now().isoformat(),
                "image_info": self._extract_image_info(image),
                "ocr_results": self._extract_text(image),
                "objects": self._detect_objects(image),
                "scene_classification": self._classify_scene(image),
                "color_analysis": self._analyze_colors(image),
                "text_summary": "",
                "search_keywords": []
            }
            
            # Generate text summary and search keywords
            analysis["text_summary"] = self._generate_text_summary(analysis)
            analysis["search_keywords"] = self._extract_search_keywords(analysis)
            
            # Cache the result
            self.analysis_cache[image_hash] = analysis
            self._save_cache()
            
            logger.info(f"✅ Analysis completed for {os.path.basename(image_path)}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing image {image_path}: {e}")
            return {"error": str(e), "image_path": image_path}
    
    def _extract_image_info(self, image):
        """Extract basic image information"""
        try:
            return {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height if image.height > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error extracting image info: {e}")
            return {}
    
    def _extract_text(self, image):
        """Extract text from image using OCR"""
        results = {
            "easyocr": [],
            "trocr": [],
            "combined_text": "",
            "text_blocks": []
        }
        
        try:
            # Convert PIL image to numpy array for EasyOCR
            image_np = np.array(image)
            
            # EasyOCR analysis
            if self.ocr_reader:
                easyocr_results = self.ocr_reader.readtext(image_np)
                results["easyocr"] = easyocr_results
                
                # Extract text blocks
                for (bbox, text, confidence) in easyocr_results:
                    results["text_blocks"].append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "method": "easyocr"
                    })
            
            # TrOCR analysis (for handwritten text)
            if self.ocr_processor and self.ocr_model:
                try:
                    # Preprocess image for TrOCR
                    pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
                    
                    # Generate text
                    with torch.no_grad():
                        generated_ids = self.ocr_model.generate(pixel_values)
                        generated_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if generated_text.strip():
                        results["trocr"] = [{
                            "text": generated_text,
                            "confidence": 0.8,  # Default confidence for TrOCR
                            "bbox": [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]],
                            "method": "trocr"
                        }]
                        
                        results["text_blocks"].append({
                            "text": generated_text,
                            "confidence": 0.8,
                            "bbox": [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]],
                            "method": "trocr"
                        })
                        
                except Exception as e:
                    logger.warning(f"TrOCR analysis failed: {e}")
            
            # Combine all text
            all_texts = []
            for block in results["text_blocks"]:
                if block["text"].strip():
                    all_texts.append(block["text"].strip())
            
            results["combined_text"] = " ".join(all_texts)
            
        except Exception as e:
            logger.error(f"Error in OCR analysis: {e}")
        
        return results
    
    def _detect_objects(self, image):
        """Detect objects in image"""
        results = {
            "objects": [],
            "object_count": 0,
            "main_objects": []
        }
        
        try:
            if self.object_processor and self.object_detector:
                # Preprocess image
                inputs = self.object_processor(images=image, return_tensors="pt")
                
                # Detect objects
                with torch.no_grad():
                    outputs = self.object_detector(**inputs)
                
                # Post-process results
                target_sizes = torch.tensor([image.size[::-1]])
                postprocessed_outputs = self.object_processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.5
                )[0]
                
                # Extract results
                for score, label, box in zip(postprocessed_outputs["scores"], postprocessed_outputs["labels"], postprocessed_outputs["boxes"]):
                    object_info = {
                        "class": self.object_detector.config.id2label[label.item()],
                        "confidence": score.item(),
                        "bbox": box.tolist(),
                        "area": (box[2] - box[0]) * (box[3] - box[1])
                    }
                    results["objects"].append(object_info)
                
                # Sort by confidence and get main objects
                results["objects"].sort(key=lambda x: x["confidence"], reverse=True)
                results["object_count"] = len(results["objects"])
                results["main_objects"] = [obj["class"] for obj in results["objects"][:5]]
                
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
        
        return results
    
    def _classify_scene(self, image):
        """Classify scene type"""
        results = {
            "scene_type": "unknown",
            "confidence": 0.0,
            "categories": []
        }
        
        try:
            if self.classifier_processor and self.image_classifier:
                # Preprocess image
                inputs = self.classifier_processor(images=image, return_tensors="pt")
                
                # Classify
                with torch.no_grad():
                    outputs = self.image_classifier(**inputs)
                
                # Get predictions
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, 5)
                
                # Extract results
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    category = self.image_classifier.config.id2label[idx.item()]
                    results["categories"].append({
                        "category": category,
                        "confidence": prob.item()
                    })
                
                # Set main scene type
                if results["categories"]:
                    results["scene_type"] = results["categories"][0]["category"]
                    results["confidence"] = results["categories"][0]["confidence"]
                
        except Exception as e:
            logger.error(f"Error in scene classification: {e}")
        
        return results
    
    def _analyze_colors(self, image):
        """Analyze dominant colors in image"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for faster processing
            small_image = image.resize((100, 100))
            pixels = np.array(small_image)
            
            # Reshape to list of pixels
            pixels = pixels.reshape(-1, 3)
            
            # Calculate dominant colors using k-means
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Count color frequencies
            color_counts = {}
            for label in labels:
                color = tuple(colors[label])
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # Sort by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "dominant_colors": [{"rgb": list(color), "frequency": count} for color, count in sorted_colors[:5]],
                "color_palette": colors.tolist(),
                "brightness": np.mean(pixels),
                "contrast": np.std(pixels)
            }
            
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return {}
    
    def _generate_text_summary(self, analysis):
        """Generate text summary from analysis results"""
        summary_parts = []
        
        # Add scene information
        if analysis.get("scene_classification", {}).get("scene_type") != "unknown":
            scene = analysis["scene_classification"]["scene_type"]
            confidence = analysis["scene_classification"]["confidence"]
            summary_parts.append(f"Scene: {scene} (confidence: {confidence:.2f})")
        
        # Add object information
        objects = analysis.get("objects", {}).get("main_objects", [])
        if objects:
            summary_parts.append(f"Objects: {', '.join(objects[:3])}")
        
        # Add text information
        combined_text = analysis.get("ocr_results", {}).get("combined_text", "")
        if combined_text:
            # Truncate long text
            if len(combined_text) > 100:
                combined_text = combined_text[:100] + "..."
            summary_parts.append(f"Text: {combined_text}")
        
        return " | ".join(summary_parts)
    
    def _extract_search_keywords(self, analysis):
        """Extract search keywords from analysis"""
        keywords = set()
        
        # Add scene keywords
        scene_type = analysis.get("scene_classification", {}).get("scene_type", "")
        if scene_type and scene_type != "unknown":
            keywords.add(scene_type.lower())
        
        # Add object keywords
        objects = analysis.get("objects", {}).get("main_objects", [])
        for obj in objects:
            keywords.add(obj.lower())
        
        # Add text keywords
        combined_text = analysis.get("ocr_results", {}).get("combined_text", "")
        if combined_text:
            # Extract meaningful words
            words = combined_text.lower().split()
            for word in words:
                if len(word) > 2 and word.isalpha():
                    keywords.add(word)
        
        # Add color keywords
        colors = analysis.get("color_analysis", {}).get("dominant_colors", [])
        for color_info in colors:
            rgb = color_info["rgb"]
            # Simple color naming
            if rgb[0] > 200 and rgb[1] > 200 and rgb[2] > 200:
                keywords.add("white")
            elif rgb[0] < 50 and rgb[1] < 50 and rgb[2] < 50:
                keywords.add("black")
            elif rgb[0] > rgb[1] and rgb[0] > rgb[2]:
                keywords.add("red")
            elif rgb[1] > rgb[0] and rgb[1] > rgb[2]:
                keywords.add("green")
            elif rgb[2] > rgb[0] and rgb[2] > rgb[1]:
                keywords.add("blue")
        
        return list(keywords)
    
    def search_by_analysis(self, query, image_analyses, top_k=20):
        """Search images based on analysis results"""
        results = []
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for image_path, analysis in image_analyses.items():
            if "error" in analysis:
                continue
            
            score = 0.0
            
            # Score based on text content
            combined_text = analysis.get("ocr_results", {}).get("combined_text", "").lower()
            if combined_text:
                for word in query_words:
                    if word in combined_text:
                        score += 2.0
            
            # Score based on objects
            objects = analysis.get("objects", {}).get("main_objects", [])
            for obj in objects:
                if obj.lower() in query_lower:
                    score += 1.5
            
            # Score based on scene
            scene = analysis.get("scene_classification", {}).get("scene_type", "").lower()
            if scene in query_lower:
                score += 1.0
            
            # Score based on keywords
            keywords = analysis.get("search_keywords", [])
            for keyword in keywords:
                if keyword in query_words:
                    score += 0.5
            
            if score > 0:
                results.append({
                    "image_path": image_path,
                    "score": score,
                    "analysis": analysis
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def batch_analyze(self, image_paths, max_workers=4):
        """Analyze multiple images in parallel"""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.analyze_image, path): path for path in image_paths}
            
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                except Exception as e:
                    logger.error(f"Error analyzing {path}: {e}")
                    results[path] = {"error": str(e)}
        
        return results
    
    def get_analysis_statistics(self):
        """Get statistics about analyzed images"""
        total_analyzed = len(self.analysis_cache)
        successful_analyses = sum(1 for analysis in self.analysis_cache.values() if "error" not in analysis)
        
        # Count text detections
        text_detections = sum(1 for analysis in self.analysis_cache.values() 
                            if "ocr_results" in analysis and analysis["ocr_results"].get("combined_text"))
        
        # Count object detections
        object_detections = sum(1 for analysis in self.analysis_cache.values() 
                              if "objects" in analysis and analysis["objects"].get("object_count", 0) > 0)
        
        return {
            "total_analyzed": total_analyzed,
            "successful_analyses": successful_analyses,
            "text_detections": text_detections,
            "object_detections": object_detections,
            "cache_size_mb": os.path.getsize(os.path.join(self.cache_dir, "analysis_cache.json")) / (1024 * 1024) if os.path.exists(os.path.join(self.cache_dir, "analysis_cache.json")) else 0
        }

# Global analyzer instance
_analyzer_instance = None

def get_analyzer():
    """Get global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = EnhancedImageAnalyzer()
    return _analyzer_instance

def analyze_single_image(image_path):
    """Analyze a single image"""
    analyzer = get_analyzer()
    return analyzer.analyze_image(image_path)

def search_images_by_analysis(query, image_paths, top_k=20):
    """Search images using analysis results"""
    analyzer = get_analyzer()
    
    # Analyze images if not already analyzed
    analyses = {}
    for path in image_paths:
        if path in analyzer.analysis_cache:
            analyses[path] = analyzer.analysis_cache[path]
        else:
            analyses[path] = analyzer.analyze_image(path)
    
    return analyzer.search_by_analysis(query, analyses, top_k)


    def _detect_landmarks(self, image_path):
        """Detect landmarks in the image"""
        try:
            return self.landmark_detector.detect_landmarks(image_path)
        except Exception as e:
            logger.error(f"Error in landmark detection: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test the analyzer
    analyzer = EnhancedImageAnalyzer()
    
    # Test with a sample image
    test_image = "static/images/sample.jpg"
    if os.path.exists(test_image):
        result = analyzer.analyze_image(test_image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Test image not found")
