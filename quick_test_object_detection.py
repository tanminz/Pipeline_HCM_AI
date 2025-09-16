#!/usr/bin/env python3
"""
Quick test script for Object Detection API
Test nhanh hiệu quả của object detection
"""

import requests
import time

def quick_test():
    """Test nhanh object detection API"""
    
    print("🚀 QUICK OBJECT DETECTION TEST")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Test với image_id = 0
    image_id = 0
    
    try:
        print(f"📸 Testing Image ID: {image_id}")
        
        # Gọi API
        start_time = time.time()
        response = requests.get(f"{base_url}/api/object_detection/{image_id}")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS!")
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"🤖 Model: {data.get('model', 'unknown')}")
            print(f"📁 Filename: {data.get('filename', 'unknown')}")
            
            # Thống kê objects
            object_count = data.get('object_count', 0)
            print(f"🔢 Total objects: {object_count}")
            
            # Object counts by class
            object_counts = data.get('object_counts', {})
            if object_counts:
                print("📊 Objects by class:")
                for obj_class, count in object_counts.items():
                    print(f"   - {obj_class}: {count}")
            
            # Main objects
            main_objects = data.get('main_objects', [])
            if main_objects:
                print(f"🎯 Main objects: {', '.join(main_objects)}")
            
            # Chi tiết objects
            objects = data.get('objects', [])
            if objects:
                print(f"\n🔍 Detected objects (showing first 5):")
                for i, obj in enumerate(objects[:5]):
                    confidence_percent = obj['confidence'] * 100
                    print(f"   {i+1}. {obj['class']} ({confidence_percent:.1f}%)")
                
                if len(objects) > 5:
                    print(f"   ... and {len(objects) - 5} more")
            
            # Đánh giá hiệu quả
            print(f"\n📈 EFFECTIVENESS ASSESSMENT:")
            
            if response_time < 2:
                print("   ⚡ Speed: EXCELLENT (< 2s)")
            elif response_time < 5:
                print("   ⚡ Speed: GOOD (2-5s)")
            else:
                print("   ⚡ Speed: SLOW (> 5s)")
            
            if object_count > 0:
                print("   🔍 Detection: WORKING (objects found)")
                
                # Kiểm tra confidence
                high_confidence_objects = [obj for obj in objects if obj['confidence'] > 0.7]
                if len(high_confidence_objects) > 0:
                    print("   🎯 Accuracy: GOOD (high confidence objects)")
                else:
                    print("   🎯 Accuracy: LOW (no high confidence objects)")
            else:
                print("   🔍 Detection: NO OBJECTS FOUND")
            
        else:
            print(f"❌ FAILED!")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_multiple_images():
    """Test với nhiều ảnh khác nhau"""
    
    print("\n🖼️ TESTING MULTIPLE IMAGES")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    test_ids = [0, 1, 2, 3, 4]
    
    results = []
    
    for image_id in test_ids:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/object_detection/{image_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                object_count = data.get('object_count', 0)
                response_time = end_time - start_time
                
                results.append({
                    'id': image_id,
                    'objects': object_count,
                    'time': response_time,
                    'success': True
                })
                
                print(f"   Image {image_id}: {object_count} objects ({response_time:.2f}s)")
            else:
                results.append({
                    'id': image_id,
                    'objects': 0,
                    'time': 0,
                    'success': False
                })
                print(f"   Image {image_id}: FAILED")
                
        except Exception as e:
            print(f"   Image {image_id}: ERROR - {e}")
            results.append({
                'id': image_id,
                'objects': 0,
                'time': 0,
                'success': False
            })
    
    # Thống kê tổng hợp
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        avg_objects = sum(r['objects'] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Successful tests: {len(successful_tests)}/{len(test_ids)}")
        print(f"   Average objects per image: {avg_objects:.1f}")
        print(f"   Average response time: {avg_time:.2f}s")

if __name__ == "__main__":
    quick_test()
    test_multiple_images()
    
    print("\n🎉 Quick test completed!")
    print("\n💡 How to check effectiveness:")
    print("   1. Response time < 5 seconds")
    print("   2. Objects detected > 0")
    print("   3. High confidence scores (> 0.7)")
    print("   4. Relevant object classes detected")



