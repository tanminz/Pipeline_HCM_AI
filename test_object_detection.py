#!/usr/bin/env python3
"""
Test script for Object Detection API
Kiểm tra hiệu quả của object detection
"""

import requests
import json
import time
from PIL import Image
import os

def test_object_detection_api():
    """Test object detection API với các ảnh khác nhau"""
    
    base_url = "http://localhost:5000"
    
    print("🔍 Testing Object Detection API...")
    print("=" * 50)
    
    # Test với một số image_id khác nhau
    test_image_ids = [0, 1, 2, 3, 4, 5]
    
    for image_id in test_image_ids:
        print(f"\n📸 Testing Image ID: {image_id}")
        print("-" * 30)
        
        try:
            # Gọi API object detection
            start_time = time.time()
            response = requests.get(f"{base_url}/api/object_detection/{image_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Success! Response time: {end_time - start_time:.2f}s")
                print(f"📊 Model used: {data.get('model', 'unknown')}")
                print(f"🔢 Total objects detected: {data.get('object_count', 0)}")
                
                # Hiển thị object counts
                object_counts = data.get('object_counts', {})
                if object_counts:
                    print("📈 Object counts by class:")
                    for obj_class, count in object_counts.items():
                        print(f"   - {obj_class}: {count}")
                
                # Hiển thị main objects
                main_objects = data.get('main_objects', [])
                if main_objects:
                    print(f"🎯 Main objects: {', '.join(main_objects)}")
                
                # Hiển thị chi tiết từng object
                objects = data.get('objects', [])
                if objects:
                    print("🔍 Detailed objects:")
                    for i, obj in enumerate(objects[:5]):  # Chỉ hiển thị 5 objects đầu
                        print(f"   {i+1}. {obj['class']} (confidence: {obj['confidence']:.2f})")
                
                if len(objects) > 5:
                    print(f"   ... and {len(objects) - 5} more objects")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()

def test_enhanced_object_detection():
    """Test enhanced object detection với thống kê chi tiết"""
    
    base_url = "http://localhost:5000"
    
    print("\n🚀 Testing Enhanced Object Detection...")
    print("=" * 50)
    
    # Test với image_id = 0
    image_id = 0
    
    try:
        print(f"📸 Testing Enhanced Object Detection for Image ID: {image_id}")
        
        # Gọi API object detection
        start_time = time.time()
        response = requests.get(f"{base_url}/api/object_detection/{image_id}")
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success! Response time: {end_time - start_time:.2f}s")
            print(f"📊 Model: {data.get('model', 'unknown')}")
            print(f"📁 Filename: {data.get('filename', 'unknown')}")
            
            # Thống kê chi tiết
            object_count = data.get('object_count', 0)
            print(f"\n📈 STATISTICS:")
            print(f"   Total objects: {object_count}")
            
            object_counts = data.get('object_counts', {})
            if object_counts:
                print(f"   Unique object types: {len(object_counts)}")
                print(f"   Most common object: {max(object_counts, key=object_counts.get) if object_counts else 'None'}")
            
            # Hiển thị tất cả objects với confidence
            objects = data.get('objects', [])
            if objects:
                print(f"\n🔍 ALL DETECTED OBJECTS ({len(objects)} total):")
                for i, obj in enumerate(objects):
                    confidence_percent = obj['confidence'] * 100
                    print(f"   {i+1:2d}. {obj['class']:15s} | Confidence: {confidence_percent:5.1f}% | Area: {obj.get('area', 0):.0f}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_performance_comparison():
    """So sánh hiệu suất giữa các lần gọi API"""
    
    base_url = "http://localhost:5000"
    
    print("\n⚡ Performance Comparison Test...")
    print("=" * 50)
    
    image_id = 0
    num_tests = 5
    
    times = []
    
    for i in range(num_tests):
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/object_detection/{image_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                times.append(response_time)
                print(f"Test {i+1}: {response_time:.3f}s")
            else:
                print(f"Test {i+1}: Failed (Status: {response.status_code})")
                
        except Exception as e:
            print(f"Test {i+1}: Exception - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Min time: {min_time:.3f}s")
        print(f"   Max time: {max_time:.3f}s")
        print(f"   Tests completed: {len(times)}/{num_tests}")

def test_different_image_types():
    """Test với các loại ảnh khác nhau"""
    
    base_url = "http://localhost:5000"
    
    print("\n🖼️ Testing Different Image Types...")
    print("=" * 50)
    
    # Test với nhiều image_id khác nhau để xem hiệu quả
    test_cases = [
        (0, "First image"),
        (10, "10th image"),
        (50, "50th image"),
        (100, "100th image"),
        (500, "500th image")
    ]
    
    for image_id, description in test_cases:
        print(f"\n📸 {description} (ID: {image_id})")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/object_detection/{image_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Success! ({end_time - start_time:.2f}s)")
                print(f"   Objects detected: {data.get('object_count', 0)}")
                
                object_counts = data.get('object_counts', {})
                if object_counts:
                    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"   Top objects: {', '.join([f'{obj}({count})' for obj, count in top_objects])}")
                
            else:
                print(f"❌ Failed (Status: {response.status_code})")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🧪 OBJECT DETECTION TEST SUITE")
    print("=" * 60)
    
    # Chạy các test
    test_object_detection_api()
    test_enhanced_object_detection()
    test_performance_comparison()
    test_different_image_types()
    
    print("\n🎉 Test suite completed!")
    print("\n💡 Tips for checking effectiveness:")
    print("   1. Response time should be reasonable (< 5 seconds)")
    print("   2. Object detection should find relevant objects")
    print("   3. Confidence scores should be meaningful (> 0.5)")
    print("   4. Object counts should make sense for the image content")



