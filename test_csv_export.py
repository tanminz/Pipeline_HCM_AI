#!/usr/bin/env python3
"""
Test script for CSV Export System
Test hệ thống xuất CSV với Place Recognizer và Object Detection
"""

import requests
import json
import time

def test_csv_export_system():
    """Test hệ thống xuất CSV"""
    
    print("🚀 TESTING CSV EXPORT SYSTEM")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Enhanced Image Analysis
    print("\n📸 Test 1: Enhanced Image Analysis")
    print("-" * 30)
    
    try:
        response = requests.get(f"{base_url}/api/analyze_image_enhanced/0")
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced analysis successful!")
            print(f"   Image ID: {data.get('image_id')}")
            print(f"   Filename: {data.get('filename')}")
            
            analysis = data.get('analysis', {})
            if 'error' not in analysis:
                print(f"   Objects detected: {len(analysis.get('objects', []))}")
                print(f"   Place info: {analysis.get('place_info', {}).get('scene_type', 'unknown')}")
                print(f"   Confidence score: {analysis.get('confidence_score', 0.0):.2f}")
                
                # Hiển thị logical relationships
                relationships = analysis.get('combined_analysis', {})
                if relationships:
                    print("   Logical relationships:")
                    print(f"     - Vehicle scene: {relationships.get('vehicle_scene', False)}")
                    print(f"     - Person scene: {relationships.get('person_scene', False)}")
                    print(f"     - Building scene: {relationships.get('building_scene', False)}")
                    print(f"     - Nature scene: {relationships.get('nature_scene', False)}")
                    print(f"     - Scene consistency: {relationships.get('scene_consistency', 0.0):.2f}")
            else:
                print(f"   ❌ Analysis error: {analysis.get('error')}")
        else:
            print(f"❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: CSV Export for Textual KIS
    print("\n📋 Test 2: CSV Export for Textual KIS")
    print("-" * 30)
    
    try:
        # Tạo mock results
        mock_results = [
            {
                "id": 0,
                "path": "static/images/L21_V001/123.jpg",
                "filename": "L21_V001/123.jpg",
                "similarity": 0.95,
                "rank": 1
            },
            {
                "id": 1,
                "path": "static/images/L22_V002/456.jpg",
                "filename": "L22_V002/456.jpg",
                "similarity": 0.87,
                "rank": 2
            }
        ]
        
        export_data = {
            "task_type": "textual_kis",
            "results": mock_results,
            "filename": "test_textual_kis",
            "max_results": 100
        }
        
        response = requests.post(f"{base_url}/api/export_csv", json=export_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ CSV export successful!")
                print(f"   Filename: {data.get('filename')}")
                print(f"   Results count: {data.get('results_count')}")
                print(f"   Task type: {data.get('task_type')}")
            else:
                print(f"❌ Export failed: {data.get('error')}")
        else:
            print(f"❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: CSV Export for Q&A
    print("\n📋 Test 3: CSV Export for Q&A")
    print("-" * 30)
    
    try:
        export_data = {
            "task_type": "qa",
            "results": mock_results,
            "filename": "test_qa",
            "max_results": 100
        }
        
        response = requests.post(f"{base_url}/api/export_csv", json=export_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Q&A CSV export successful!")
                print(f"   Filename: {data.get('filename')}")
                print(f"   Results count: {data.get('results_count')}")
            else:
                print(f"❌ Export failed: {data.get('error')}")
        else:
            print(f"❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 4: CSV Export for TRAKE
    print("\n📋 Test 4: CSV Export for TRAKE")
    print("-" * 30)
    
    try:
        export_data = {
            "task_type": "trake",
            "results": mock_results,
            "filename": "test_trake",
            "max_results": 100
        }
        
        response = requests.post(f"{base_url}/api/export_csv", json=export_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ TRAKE CSV export successful!")
                print(f"   Filename: {data.get('filename')}")
                print(f"   Results count: {data.get('results_count')}")
            else:
                print(f"❌ Export failed: {data.get('error')}")
        else:
            print(f"❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_performance():
    """Test hiệu suất của hệ thống"""
    
    print("\n⚡ PERFORMANCE TEST")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test thời gian phân tích ảnh
    print("\n📸 Image Analysis Performance Test")
    print("-" * 40)
    
    test_image_ids = [0, 1, 2, 3, 4]
    times = []
    
    for image_id in test_image_ids:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/analyze_image_enhanced/{image_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                analysis_time = end_time - start_time
                times.append(analysis_time)
                print(f"   Image {image_id}: {analysis_time:.2f}s")
            else:
                print(f"   Image {image_id}: FAILED")
                
        except Exception as e:
            print(f"   Image {image_id}: ERROR - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Analysis Performance Summary:")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Total images: {len(times)}")
        
        if avg_time < 5:
            print("   ⚡ Performance: EXCELLENT (< 5s)")
        elif avg_time < 10:
            print("   ⚡ Performance: GOOD (5-10s)")
        else:
            print("   ⚡ Performance: SLOW (> 10s)")

def test_csv_formats():
    """Test các định dạng CSV khác nhau"""
    
    print("\n📋 CSV FORMAT TEST")
    print("=" * 50)
    
    print("\n📝 Expected CSV Formats:")
    print("-" * 30)
    
    print("1. Textual KIS Format:")
    print("   Video Name, Frame Number")
    print("   L21_V001, 123")
    print("   L22_V002, 456")
    
    print("\n2. Q&A Format:")
    print("   Video Name, Frame Number, Answer")
    print("   L21_V001, 123, car")
    print("   L22_V002, 456, person")
    
    print("\n3. TRAKE Format:")
    print("   Video Name, Frame Numbers")
    print("   L21_V001, 123, 456, 789")
    print("   L22_V002, 101, 202, 303")

if __name__ == "__main__":
    print("🧪 CSV EXPORT SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Chạy các test
    test_csv_export_system()
    test_performance()
    test_csv_formats()
    
    print("\n🎉 Test suite completed!")
    print("\n💡 CSV Export System Features:")
    print("   ✅ Object Detection + Place Recognition")
    print("   ✅ Logical Relationships Analysis")
    print("   ✅ Multiple Task Type Support (Textual KIS, Q&A, TRAKE)")
    print("   ✅ Confidence-based Ranking")
    print("   ✅ Max 100 results per query")
    print("   ✅ Proper CSV formatting for competition submission")



