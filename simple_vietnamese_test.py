#!/usr/bin/env python3
"""
Simple Vietnamese Query Test
Test đơn giản cho câu query tiếng Việt
"""

import requests
import time

def test_simple_vietnamese_query():
    """Test câu query tiếng Việt đơn giản"""
    
    print("🇻🇳 SIMPLE VIETNAMESE QUERY TEST")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test câu query tiếng Việt đơn giản
    test_queries = [
        "xe ô tô",
        "người đi bộ", 
        "tòa nhà",
        "cây xanh",
        "đường phố"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        
        try:
            start_time = time.time()
            
            # Test API endpoint
            response = requests.get(f"{base_url}/api/search", params={
                'q': query,
                'k': 10
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"   Status: {response.status_code}")
            print(f"   Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    results_count = len(data.get('results', []))
                    print(f"   ✅ Found {results_count} results")
                    
                    if results_count > 0:
                        # Hiển thị kết quả đầu tiên
                        first_result = data['results'][0]
                        filename = first_result.get('filename', 'unknown')
                        similarity = first_result.get('similarity', 0.0)
                        print(f"   📊 First result: {filename} (similarity: {similarity:.3f})")
                        
                except Exception as e:
                    print(f"   ❌ JSON parse error: {e}")
                    print(f"   Response text: {response.text[:200]}...")
            else:
                print(f"   ❌ HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_web_interface():
    """Test web interface với câu query tiếng Việt"""
    
    print("\n🌐 WEB INTERFACE TEST")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test web interface
    test_query = "xe ô tô"
    
    try:
        print(f"🔍 Testing web interface with query: '{test_query}'")
        
        response = requests.get(f"{base_url}/textsearch", params={
            'textquery': test_query,
            'k': 10,
            'per_page': 10
        })
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                results_count = len(data.get('results', []))
                print(f"   ✅ Found {results_count} results")
                
                if results_count > 0:
                    print("   📊 Sample results:")
                    for i, result in enumerate(data['results'][:3], 1):
                        filename = result.get('filename', 'unknown')
                        similarity = result.get('similarity', 0.0)
                        print(f"      {i}. {filename} (similarity: {similarity:.3f})")
                        
            except Exception as e:
                print(f"   ❌ JSON parse error: {e}")
                print(f"   Response text: {response.text[:200]}...")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def test_server_status():
    """Test trạng thái server"""
    
    print("\n🔧 SERVER STATUS TEST")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status")
        print(f"Status endpoint: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Server is running")
                print(f"   Total images: {data.get('total_images', 'unknown')}")
                print(f"   Status: {data.get('status', 'unknown')}")
            except Exception as e:
                print(f"❌ JSON parse error: {e}")
        else:
            print(f"❌ Server error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("🇻🇳 VIETNAMESE QUERY TESTING")
    print("=" * 60)
    
    # Test server status
    test_server_status()
    
    # Test simple queries
    test_simple_vietnamese_query()
    
    # Test web interface
    test_web_interface()
    
    print("\n�� Test completed!")



