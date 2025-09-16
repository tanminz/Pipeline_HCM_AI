#!/usr/bin/env python3
"""
Test Nearby Frames Search Functionality
"""

import json
import logging
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_nearby_frames():
    """Test nearby frames search functionality"""
    
    base_url = "http://localhost:5000/api/search_nearby_frames"
    
    # Test cases for nearby frames
    test_cases = [
        {
            'frame_id': 'L21_V001/152.jpg',
            'video_name': 'L21_V001',
            'k': 10,
            'description': 'Test frame in middle of video'
        },
        {
            'frame_id': 'L21_V001/001.jpg',
            'video_name': 'L21_V001', 
            'k': 5,
            'description': 'Test frame at beginning of video'
        },
        {
            'frame_id': 'L22_V015/294.jpg',
            'video_name': 'L22_V015',
            'k': 8,
            'description': 'Test frame in different video'
        },
        {
            'frame_id': 'L25_V069/155.jpg',
            'video_name': 'L25_V069',
            'k': 12,
            'description': 'Test frame with more nearby frames'
        }
    ]
    
    logger.info("🚀 Testing Nearby Frames Search...")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{i}. Testing: {test_case['description']}")
        logger.info(f"   Frame: {test_case['frame_id']}")
        logger.info(f"   Video: {test_case['video_name']}")
        logger.info(f"   K: {test_case['k']}")
        
        try:
            # Make API request
            params = {
                'frame_id': test_case['frame_id'],
                'video_name': test_case['video_name'],
                'k': test_case['k']
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                results = data.get('results', [])
                target_frame = data.get('target_frame', 0)
                video_name = data.get('video_name', '')
                nearby_range = data.get('nearby_range', '')
                
                logger.info(f"✅ Found {count} nearby frames")
                logger.info(f"   Target frame: {target_frame}")
                logger.info(f"   Video: {video_name}")
                logger.info(f"   Range: {nearby_range}")
                
                if results:
                    logger.info("Nearby frames:")
                    for j, result in enumerate(results[:5], 1):
                        filename = result.get('filename', 'Unknown')
                        frame_number = result.get('frame_number', 0)
                        distance = result.get('distance', 0)
                        is_target = result.get('is_target', False)
                        
                        target_marker = " [TARGET]" if is_target else ""
                        logger.info(f"  {j}. {filename} (frame {frame_number}, distance {distance}){target_marker}")
                    
                    if len(results) > 5:
                        logger.info(f"  ... and {len(results) - 5} more frames")
                else:
                    logger.info("  No nearby frames found")
                    
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                error_data = response.json() if response.content else {}
                logger.error(f"   Error: {error_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Error testing nearby frames: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    logger.info("\n✅ Nearby frames testing completed!")

def test_complex_vietnamese_queries():
    """Test complex Vietnamese queries with enhanced translation"""
    
    base_url = "http://localhost:5000/api/search"
    
    # Complex Vietnamese queries to test enhanced translation
    complex_queries = [
        "Một người đàn ông đang trả lời phỏng vấn trong một lễ hội. Phía sau người đàn ông này là một vật trang trí có hình dáng con chim màu tím.",
        "Hai chàng trai đang trượt tuyết trên núi cao",
        "Các cô gái mặc áo dài hoạt tiết sọc caro đang nhảy múa",
        "Con trâu đang ăn cỏ trong đồng lúa xanh",
        "Xe hơi đỏ chạy trên đường phố đông đúc",
        "Tòa nhà cao tầng trong thành phố hiện đại",
        "Cây cối xanh tươi trong thiên nhiên hoang dã",
        "Máy bay trắng bay trên bầu trời xanh",
        "Thuyền lớn neo đậu ở cảng biển",
        "Người đàn ông đang làm việc trên máy tính để bàn"
    ]
    
    logger.info("\n🌐 Testing Complex Vietnamese Queries with Enhanced Translation...")
    
    for i, query in enumerate(complex_queries, 1):
        logger.info(f"\n{i}. Testing complex query:")
        logger.info(f"   '{query}'")
        
        try:
            # Make API request
            response = requests.get(f"{base_url}?q={query}&k=5", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                results = data.get('results', [])
                engine_type = data.get('engine_type', 'unknown')
                
                logger.info(f"✅ Found {count} results (Engine: {engine_type})")
                
                if results:
                    logger.info("Top 3 results:")
                    for j, result in enumerate(results[:3], 1):
                        filename = result.get('filename', 'Unknown')
                        similarity = result.get('similarity', 0)
                        logger.info(f"  {j}. {filename} (similarity: {similarity:.3f})")
                else:
                    logger.info("  No results found")
                    
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing query: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    logger.info("\n✅ Complex Vietnamese queries testing completed!")

if __name__ == "__main__":
    test_nearby_frames()
    test_complex_vietnamese_queries()


