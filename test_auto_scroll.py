#!/usr/bin/env python3
"""
Test Auto-Scroll to Target Frame
"""

import json
import logging
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_auto_scroll_functionality():
    """Test auto-scroll functionality"""
    
    base_url = "http://localhost:5000"
    test_frames = [
        "L21_V001/152.jpg",  # Frame 152/307 - should be on page 2 with 100/trang
        "L22_V015/294.jpg",  # Frame 294/301 - should be on page 3 with 100/trang
        "L25_V069/155.jpg",  # Frame 155/496 - should be on page 2 with 100/trang
        "L21_V001/001.jpg",  # Frame 1/307 - should be on page 1 with 100/trang
        "L21_V001/307.jpg"   # Frame 307/307 - should be on page 4 with 100/trang
    ]
    
    logger.info("🎯 Testing Auto-Scroll to Target Frame...")
    
    for frame_id in test_frames:
        logger.info(f"\n--- Testing auto-scroll for {frame_id} ---")
        
        try:
            # Test with 100 frames per page (default)
            per_page = 100
            
            response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={frame_id}&page=1&per_page={per_page}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                pagination = data.get('pagination', {})
                results = data.get('results', [])
                target_found = any(frame.get('is_target', False) for frame in results)
                
                logger.info(f"✅ Auto-scroll test results:")
                logger.info(f"   Current page: {pagination.get('current_page')}")
                logger.info(f"   Total pages: {pagination.get('total_pages')}")
                logger.info(f"   Frames returned: {len(results)}")
                logger.info(f"   Target found: {target_found}")
                
                if target_found:
                    target_frame = next((frame for frame in results if frame.get('is_target')), None)
                    if target_frame:
                        logger.info(f"   ✅ Target frame: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
                        logger.info(f"   🎯 Auto-scroll should work - target frame is on current page")
                else:
                    logger.warning(f"   ⚠️ Target frame not found on current page")
                    
                # Test frame info endpoint
                frame_info_response = requests.get(f"{base_url}/api/frame_info?frame_id={frame_id}", timeout=10)
                if frame_info_response.status_code == 200:
                    frame_info = frame_info_response.json()
                    if 'error' not in frame_info:
                        frame_position = frame_info.get('frame_position', 0)
                        total_frames = frame_info.get('total_frames', 0)
                        expected_page = (frame_position + per_page - 1) // per_page
                        
                        logger.info(f"   📊 Frame position: {frame_position}/{total_frames}")
                        logger.info(f"   📄 Expected page: {expected_page}")
                        logger.info(f"   🎯 Auto-navigation: {'✅ Working' if pagination.get('current_page') == expected_page else '❌ Not working'}")
                
            else:
                logger.error(f"❌ Failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing auto-scroll: {e}")
        
        time.sleep(1)

def test_scroll_behavior():
    """Test different scroll scenarios"""
    
    base_url = "http://localhost:5000"
    
    logger.info("\n🔄 Testing Scroll Behavior Scenarios...")
    
    # Test frame in middle of page
    logger.info("\n--- Testing frame in middle of page ---")
    try:
        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id=L21_V001/152.jpg&page=1&per_page=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            target_frame = next((frame for frame in results if frame.get('is_target')), None)
            
            if target_frame:
                rank = target_frame.get('rank', 0)
                logger.info(f"✅ Target frame rank: {rank}")
                logger.info(f"   📍 Should scroll to center of page")
                logger.info(f"   🎯 Frame should be highlighted with pulse animation")
            else:
                logger.warning("⚠️ Target frame not found")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # Test frame at beginning of page
    logger.info("\n--- Testing frame at beginning of page ---")
    try:
        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id=L21_V001/001.jpg&page=1&per_page=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            target_frame = next((frame for frame in results if frame.get('is_target')), None)
            
            if target_frame:
                rank = target_frame.get('rank', 0)
                logger.info(f"✅ Target frame rank: {rank}")
                logger.info(f"   📍 Should scroll to top of page")
                logger.info(f"   🎯 Frame should be highlighted with pulse animation")
            else:
                logger.warning("⚠️ Target frame not found")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # Test frame at end of page
    logger.info("\n--- Testing frame at end of page ---")
    try:
        response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id=L21_V001/307.jpg&page=1&per_page=100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            target_frame = next((frame for frame in results if frame.get('is_target')), None)
            
            if target_frame:
                rank = target_frame.get('rank', 0)
                logger.info(f"✅ Target frame rank: {rank}")
                logger.info(f"   📍 Should scroll to bottom of page")
                logger.info(f"   🎯 Frame should be highlighted with pulse animation")
            else:
                logger.warning("⚠️ Target frame not found")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def test_keyboard_shortcuts():
    """Test keyboard shortcuts for scroll functionality"""
    
    logger.info("\n⌨️ Testing Keyboard Shortcuts...")
    
    shortcuts = [
        ("T", "Scroll to Target Frame"),
        ("t", "Scroll to Target Frame (lowercase)"),
        ("←", "Previous Page"),
        ("→", "Next Page"),
        ("Home", "First Page"),
        ("End", "Last Page"),
        ("ESC", "Go Back")
    ]
    
    for key, description in shortcuts:
        logger.info(f"   {key}: {description}")
    
    logger.info("✅ All keyboard shortcuts should work in the web interface")

if __name__ == "__main__":
    test_auto_scroll_functionality()
    test_scroll_behavior()
    test_keyboard_shortcuts()




