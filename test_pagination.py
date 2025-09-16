#!/usr/bin/env python3
"""
Test Pagination Functionality for Nearby Frames
"""

import json
import logging
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pagination():
    """Test pagination functionality"""
    
    base_url = "http://localhost:5000"
    test_frame = "L21_V001/152.jpg"
    
    logger.info("📄 Testing Pagination Functionality...")
    
    # Test different page sizes
    page_sizes = [10, 20, 30, 50]
    
    for per_page in page_sizes:
        logger.info(f"\n--- Testing with {per_page} frames per page ---")
        
        # Test first page
        try:
            response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={test_frame}&page=1&per_page={per_page}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                pagination = data.get('pagination', {})
                
                logger.info(f"✅ Page 1: {data.get('count')} frames")
                logger.info(f"   Total pages: {pagination.get('total_pages')}")
                logger.info(f"   Range: {pagination.get('start_index')}-{pagination.get('end_index')}")
                logger.info(f"   Has next: {pagination.get('has_next')}")
                logger.info(f"   Has prev: {pagination.get('has_prev')}")
                
                # Test second page if available
                if pagination.get('has_next'):
                    response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={test_frame}&page=2&per_page={per_page}", timeout=10)
                    if response.status_code == 200:
                        data2 = response.json()
                        pagination2 = data2.get('pagination', {})
                        
                        logger.info(f"✅ Page 2: {data2.get('count')} frames")
                        logger.info(f"   Range: {pagination2.get('start_index')}-{pagination2.get('end_index')}")
                        logger.info(f"   Has next: {pagination2.get('has_next')}")
                        logger.info(f"   Has prev: {pagination2.get('has_prev')}")
                    else:
                        logger.error(f"❌ Page 2 failed with status {response.status_code}")
                
                # Test last page
                total_pages = pagination.get('total_pages', 1)
                if total_pages > 1:
                    response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={test_frame}&page={total_pages}&per_page={per_page}", timeout=10)
                    if response.status_code == 200:
                        data_last = response.json()
                        pagination_last = data_last.get('pagination', {})
                        
                        logger.info(f"✅ Last page ({total_pages}): {data_last.get('count')} frames")
                        logger.info(f"   Range: {pagination_last.get('start_index')}-{pagination_last.get('end_index')}")
                        logger.info(f"   Has next: {pagination_last.get('has_next')}")
                        logger.info(f"   Has prev: {pagination_last.get('has_prev')}")
                    else:
                        logger.error(f"❌ Last page failed with status {response.status_code}")
                
            else:
                logger.error(f"❌ Failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing pagination: {e}")
        
        time.sleep(1)

def test_target_frame_navigation():
    """Test navigation to target frame"""
    
    base_url = "http://localhost:5000"
    test_frames = [
        "L21_V001/152.jpg",  # Frame 152
        "L22_V015/294.jpg",  # Frame 294
        "L25_V069/155.jpg"   # Frame 155
    ]
    
    logger.info("\n🎯 Testing Target Frame Navigation...")
    
    for frame_id in test_frames:
        logger.info(f"\n--- Testing navigation for {frame_id} ---")
        
        try:
            # Get frame info first
            response = requests.get(f"{base_url}/api/frame_info?frame_id={frame_id}", timeout=10)
            if response.status_code == 200:
                frame_data = response.json()
                if 'error' not in frame_data:
                    frame_position = frame_data.get('frame_position', 0)
                    total_frames = frame_data.get('total_frames', 0)
                    
                    logger.info(f"✅ Frame info: Position {frame_position}/{total_frames}")
                    
                    # Test with different page sizes to see which page contains target
                    for per_page in [10, 20, 30]:
                        target_page = (frame_position + per_page - 1) // per_page
                        
                        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page={target_page}&per_page={per_page}", timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get('results', [])
                            
                            # Check if target frame is in this page
                            target_found = any(frame.get('is_target', False) for frame in results)
                            
                            logger.info(f"   Per page {per_page}: Target page {target_page}, Target found: {target_found}")
                            
                            if target_found:
                                target_frame = next((frame for frame in results if frame.get('is_target')), None)
                                if target_frame:
                                    logger.info(f"     Target frame: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
                        else:
                            logger.error(f"❌ Failed to get page {target_page} with per_page {per_page}")
                    
                else:
                    logger.error(f"❌ Frame info error: {frame_data.get('error')}")
            else:
                logger.error(f"❌ Frame info failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing target navigation: {e}")
        
        time.sleep(1)

def test_complete_workflow():
    """Test complete workflow with pagination"""
    
    base_url = "http://localhost:5000"
    
    logger.info("\n🔄 Testing Complete Workflow with Pagination...")
    
    # Step 1: Search for a query
    logger.info("\nStep 1: Searching for 'person'...")
    try:
        response = requests.get(f"{base_url}/api/search?q=person&k=5", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            logger.info(f"✅ Found {len(results)} results")
            
            if results:
                # Step 2: Get first result and test pagination
                first_result = results[0]
                frame_id = first_result.get('filename')
                logger.info(f"\nStep 2: Testing pagination for {frame_id}...")
                
                # Test with 20 frames per page
                per_page = 20
                response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page=1&per_page={per_page}", timeout=10)
                if response.status_code == 200:
                    nearby_data = response.json()
                    pagination = nearby_data.get('pagination', {})
                    
                    logger.info(f"✅ Pagination info:")
                    logger.info(f"   Total frames: {nearby_data.get('total_frames')}")
                    logger.info(f"   Total pages: {pagination.get('total_pages')}")
                    logger.info(f"   Current page: {pagination.get('current_page')}")
                    logger.info(f"   Frames on this page: {nearby_data.get('count')}")
                    logger.info(f"   Range: {pagination.get('start_index')}-{pagination.get('end_index')}")
                    
                    # Check if target frame is on this page
                    results = nearby_data.get('results', [])
                    target_frame = next((frame for frame in results if frame.get('is_target')), None)
                    
                    if target_frame:
                        logger.info(f"✅ Target frame found on page {pagination.get('current_page')}")
                        logger.info(f"   Target: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
                    else:
                        logger.info(f"⚠️ Target frame not on page {pagination.get('current_page')}")
                        
                        # Find which page contains target
                        frame_info_response = requests.get(f"{base_url}/api/frame_info?frame_id={frame_id}", timeout=10)
                        if frame_info_response.status_code == 200:
                            frame_info = frame_info_response.json()
                            target_position = frame_info.get('frame_position', 0)
                            target_page = (target_position + per_page - 1) // per_page
                            
                            logger.info(f"   Target should be on page {target_page}")
                            
                            # Test target page
                            if target_page != 1:
                                response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page={target_page}&per_page={per_page}", timeout=10)
                                if response.status_code == 200:
                                    target_page_data = response.json()
                                    target_page_results = target_page_data.get('results', [])
                                    target_found = any(frame.get('is_target', False) for frame in target_page_results)
                                    
                                    logger.info(f"✅ Target page {target_page}: Target found = {target_found}")
                    
                    logger.info("✅ Complete workflow successful!")
                else:
                    logger.error(f"❌ Nearby frames failed with status {response.status_code}")
            else:
                logger.error("❌ No search results found")
        else:
            logger.error(f"❌ Search failed with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Error in workflow: {e}")

if __name__ == "__main__":
    test_pagination()
    test_target_frame_navigation()
    test_complete_workflow()




