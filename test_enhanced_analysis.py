#!/usr/bin/env python3
"""
Test Enhanced Image Analysis for HCMC AI Challenge
Tests OCR, object detection, and image analysis capabilities
"""

import os
import json
import time
from PIL import Image
import numpy as np

def test_enhanced_analyzer():
    """Test the enhanced image analyzer"""
    print("🧪 Testing Enhanced Image Analyzer...")
    
    try:
        from enhanced_image_analyzer import EnhancedImageAnalyzer
        
        # Initialize analyzer
        print("🔄 Initializing Enhanced Image Analyzer...")
        analyzer = EnhancedImageAnalyzer()
        print("✅ Enhanced Image Analyzer initialized successfully")
        
        # Test with sample images
        test_images = []
        
        # Look for test images in static/images
        static_dir = "static/images"
        if os.path.exists(static_dir):
            for filename in os.listdir(static_dir)[:5]:  # Test first 5 images
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(static_dir, filename))
        
        if not test_images:
            print("⚠️ No test images found in static/images")
            return
        
        print(f"🔍 Found {len(test_images)} test images")
        
        # Test analysis on each image
        for i, image_path in enumerate(test_images):
            print(f"\n📸 Testing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
            
            try:
                # Analyze image
                start_time = time.time()
                analysis = analyzer.analyze_image(image_path)
                end_time = time.time()
                
                if "error" in analysis:
                    print(f"❌ Analysis failed: {analysis['error']}")
                    continue
                
                print(f"⏱️ Analysis time: {end_time - start_time:.2f} seconds")
                
                # Display results
                print(f"📊 Image Info: {analysis.get('image_info', {}).get('size', 'Unknown')}")
                
                # OCR results
                ocr_results = analysis.get('ocr_results', {})
                combined_text = ocr_results.get('combined_text', '')
                if combined_text:
                    print(f"📝 Extracted Text: {combined_text[:100]}...")
                else:
                    print("📝 No text detected")
                
                # Object detection
                objects = analysis.get('objects', {})
                main_objects = objects.get('main_objects', [])
                if main_objects:
                    print(f"🎯 Detected Objects: {', '.join(main_objects[:5])}")
                else:
                    print("🎯 No objects detected")
                
                # Scene classification
                scene = analysis.get('scene_classification', {})
                scene_type = scene.get('scene_type', 'unknown')
                confidence = scene.get('confidence', 0.0)
                if scene_type != 'unknown':
                    print(f"🏞️ Scene: {scene_type} (confidence: {confidence:.2f})")
                else:
                    print("🏞️ Scene: Unknown")
                
                # Search keywords
                keywords = analysis.get('search_keywords', [])
                if keywords:
                    print(f"🔍 Search Keywords: {', '.join(keywords[:10])}")
                
                # Text summary
                summary = analysis.get('text_summary', '')
                if summary:
                    print(f"📋 Summary: {summary}")
                
            except Exception as e:
                print(f"❌ Error analyzing {image_path}: {e}")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        test_queries = ["person", "car", "text", "outdoor", "indoor"]
        
        for query in test_queries:
            print(f"\n🔎 Testing search for: '{query}'")
            try:
                results = analyzer.search_by_analysis(query, test_images, top_k=3)
                print(f"✅ Found {len(results)} results for '{query}'")
                
                for i, result in enumerate(results[:2]):  # Show top 2 results
                    score = result['score']
                    analysis = result['analysis']
                    summary = analysis.get('text_summary', 'No summary')
                    print(f"  {i+1}. Score: {score:.2f} - {summary[:50]}...")
                    
            except Exception as e:
                print(f"❌ Search failed for '{query}': {e}")
        
        # Get statistics
        stats = analyzer.get_analysis_statistics()
        print(f"\n📊 Analyzer Statistics:")
        print(f"  Total analyzed: {stats.get('total_analyzed', 0)}")
        print(f"  Successful analyses: {stats.get('successful_analyses', 0)}")
        print(f"  Text detections: {stats.get('text_detections', 0)}")
        print(f"  Object detections: {stats.get('object_detections', 0)}")
        print(f"  Cache size: {stats.get('cache_size_mb', 0):.2f} MB")
        
        print("\n🎉 Enhanced Image Analyzer test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Failed to import Enhanced Image Analyzer: {e}")
        print("💡 Make sure to install dependencies first:")
        print("   python install_enhanced_dependencies.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_ocr_only():
    """Test OCR functionality only"""
    print("\n🧪 Testing OCR functionality...")
    
    try:
        import easyocr
        
        # Initialize EasyOCR
        print("🔄 Initializing EasyOCR...")
        reader = easyocr.Reader(['en', 'vi'])
        print("✅ EasyOCR initialized successfully")
        
        # Test with a sample image
        test_image = "static/images/sample.jpg"
        if os.path.exists(test_image):
            print(f"🔍 Testing OCR on: {test_image}")
            
            # Read text
            results = reader.readtext(test_image)
            
            if results:
                print(f"✅ Found {len(results)} text blocks:")
                for i, (bbox, text, confidence) in enumerate(results):
                    print(f"  {i+1}. Text: '{text}' (confidence: {confidence:.2f})")
            else:
                print("📝 No text detected")
        else:
            print("⚠️ Sample image not found for OCR test")
            
    except ImportError:
        print("❌ EasyOCR not installed. Install with: pip install easyocr")
    except Exception as e:
        print(f"❌ OCR test failed: {e}")

def test_object_detection():
    """Test object detection functionality"""
    print("\n🧪 Testing Object Detection...")
    
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        import torch
        
        # Initialize model
        print("🔄 Loading object detection model...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        print("✅ Object detection model loaded successfully")
        
        # Test with a sample image
        test_image = "static/images/sample.jpg"
        if os.path.exists(test_image):
            print(f"🔍 Testing object detection on: {test_image}")
            
            # Load and process image
            image = Image.open(test_image)
            inputs = processor(images=image, return_tensors="pt")
            
            # Detect objects
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            # Display results
            if len(results["scores"]) > 0:
                print(f"✅ Detected {len(results['scores'])} objects:")
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    object_name = model.config.id2label[label.item()]
                    confidence = score.item()
                    print(f"  - {object_name} (confidence: {confidence:.2f})")
            else:
                print("🎯 No objects detected")
        else:
            print("⚠️ Sample image not found for object detection test")
            
    except ImportError:
        print("❌ Transformers not installed. Install with: pip install transformers")
    except Exception as e:
        print(f"❌ Object detection test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Enhanced Image Analysis Test Suite")
    print("=" * 50)
    
    # Test OCR
    test_ocr_only()
    
    # Test object detection
    test_object_detection()
    
    # Test full analyzer
    test_enhanced_analyzer()
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")

if __name__ == "__main__":
    main()






