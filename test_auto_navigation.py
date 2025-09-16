#!/usr/bin/env python3
"""
Test Auto-Navigation to Target Frame
"""

import json
import logging
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_auto_navigation():
    """Test auto-navigation to target frame"""
    
    base_url = "http://localhost:5000"
    test_frames = [
        "L21_V001/152.jpg",  # Frame 152/307
        "L22_V015/294.jpg",  # Frame 294/301  
        "L25_V069/155.jpg",  # Frame 155/496
        "L21_V001/001.jpg",  # Frame 1/307 (first frame)
        "L21_V001/307.jpg"   # Frame 307/307 (last frame)
    ]
    
    logger.info("🎯 Testing Auto-Navigation to Target Frame...")
    
    for frame_id in test_frames:
        logger.info(f"\n--- Testing auto-navigation for {frame_id} ---")
        
        try:
            # Get frame info first
            response = requests.get(f"{base_url}/api/frame_info?frame_id={frame_id}", timeout=10)
            if response.status_code == 200:
                frame_data = response.json()
                if 'error' not in frame_data:
                    frame_position = frame_data.get('frame_position', 0)
                    total_frames = frame_data.get('total_frames', 0)
                    
                    logger.info(f"✅ Frame info: Position {frame_position}/{total_frames}")
                    
                    # Test auto-navigation with different page sizes
                    for per_page in [20, 50, 100, 200]:
                        expected_page = (frame_position + per_page - 1) // per_page
                        
                        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page=1&per_page={per_page}", timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            pagination = data.get('pagination', {})
                            actual_page = pagination.get('current_page', 1)
                            target_page = data.get('target_page', 0)
                            
                            # Check if target frame is on the returned page
                            results = data.get('results', [])
                            target_found = any(frame.get('is_target', False) for frame in results)
                            
                            logger.info(f"   Per page {per_page}:")
                            logger.info(f"     Expected page: {expected_page}")
                            logger.info(f"     Actual page: {actual_page}")
                            logger.info(f"     Target page: {target_page}")
                            logger.info(f"     Target found: {target_found}")
                            logger.info(f"     Frames returned: {len(results)}")
                            
                            if target_found:
                                target_frame = next((frame for frame in results if frame.get('is_target')), None)
                                if target_frame:
                                    logger.info(f"     ✅ Target frame: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
                            else:
                                logger.warning(f"     ⚠️ Target frame not found on page {actual_page}")
                                
                        else:
                            logger.error(f"❌ Failed to get data with per_page {per_page}")
                    
                else:
                    logger.error(f"❌ Frame info error: {frame_data.get('error')}")
            else:
                logger.error(f"❌ Frame info failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing auto-navigation: {e}")
        
        time.sleep(1)

def test_edge_cases():
    """Test edge cases for auto-navigation"""
    
    base_url = "http://localhost:5000"
    
    logger.info("\n🔍 Testing Edge Cases...")
    
    # Test first frame
    logger.info("\n--- Testing first frame ---")
    try:
        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id=L21_V001/001.jpg&page=1&per_page=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            pagination = data.get('pagination', {})
            target_found = any(frame.get('is_target', False) for frame in data.get('results', []))
            
            logger.info(f"✅ First frame test:")
            logger.info(f"   Current page: {pagination.get('current_page')}")
            logger.info(f"   Target found: {target_found}")
            logger.info(f"   Should be on page 1: {pagination.get('current_page') == 1}")
        else:
            logger.error(f"❌ First frame test failed")
    except Exception as e:
        logger.error(f"❌ Error testing first frame: {e}")
    
    # Test last frame
    logger.info("\n--- Testing last frame ---")
    try:
        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id=L21_V001/307.jpg&page=1&per_page=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            pagination = data.get('pagination', {})
            target_found = any(frame.get('is_target', False) for frame in data.get('results', []))
            
            logger.info(f"✅ Last frame test:")
            logger.info(f"   Current page: {pagination.get('current_page')}")
            logger.info(f"   Total pages: {pagination.get('total_pages')}")
            logger.info(f"   Target found: {target_found}")
            logger.info(f"   Should be on last page: {pagination.get('current_page') == pagination.get('total_pages')}")
        else:
            logger.error(f"❌ Last frame test failed")
    except Exception as e:
        logger.error(f"❌ Error testing last frame: {e}")

def test_performance():
    """Test performance with 100 frames per page"""
    
    base_url = "http://localhost:5000"
    
    logger.info("\n⚡ Testing Performance with 100 frames/trang...")
    
    test_frames = [
        "L21_V001/152.jpg",
        "L22_V015/294.jpg", 
        "L25_V069/155.jpg"
    ]
    
    for frame_id in test_frames:
        logger.info(f"\n--- Performance test for {frame_id} ---")
        
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page=1&per_page=100", timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                pagination = data.get('pagination', {})
                results = data.get('results', [])
                target_found = any(frame.get('is_target', False) for frame in results)
                
                logger.info(f"✅ Performance results:")
                logger.info(f"   Response time: {end_time - start_time:.2f} seconds")
                logger.info(f"   Frames returned: {len(results)}")
                logger.info(f"   Current page: {pagination.get('current_page')}")
                logger.info(f"   Total pages: {pagination.get('total_pages')}")
                logger.info(f"   Target found: {target_found}")
                
                if target_found:
                    target_frame = next((frame for frame in results if frame.get('is_target')), None)
                    if target_frame:
                        logger.info(f"   Target frame: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
            else:
                logger.error(f"❌ Performance test failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error in performance test: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_auto_navigation()
    test_edge_cases()
    test_performance()




