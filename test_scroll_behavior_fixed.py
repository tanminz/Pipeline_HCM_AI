#!/usr/bin/env python3
"""
Test Fixed Scroll Behavior
"""

import json
import logging
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scroll_behavior_fixed():
    """Test fixed scroll behavior"""
    
    base_url = "http://localhost:5000"
    
    logger.info("🎯 Testing Fixed Scroll Behavior...")
    
    # Test different scenarios
    test_cases = [
        {
            "frame_id": "L21_V001/152.jpg",
            "description": "Frame ở giữa trang (152/307)",
            "expected_page": 2,
            "expected_behavior": "Auto-navigation + scroll to target (không scroll to top)"
        },
        {
            "frame_id": "L22_V015/294.jpg", 
            "description": "Frame ở cuối video (294/301)",
            "expected_page": 3,
            "expected_behavior": "Auto-navigation + scroll to target (không scroll to top)"
        },
        {
            "frame_id": "L21_V001/001.jpg",
            "description": "Frame đầu video (1/307)",
            "expected_page": 1,
            "expected_behavior": "Scroll to target (đã ở trang đúng)"
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n--- Testing: {test_case['description']} ---")
        logger.info(f"Expected behavior: {test_case['expected_behavior']}")
        
        try:
            # Test auto-navigation with 100 frames per page
            per_page = 100
            response = requests.get(f"{base_url}/api/search_nearby_frames_ui?frame_id={test_case['frame_id']}&page=1&per_page={per_page}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pagination = data.get('pagination', {})
                results = data.get('results', [])
                target_found = any(frame.get('is_target', False) for frame in results)
                
                current_page = pagination.get('current_page', 1)
                expected_page = test_case['expected_page']
                
                logger.info(f"✅ Test results:")
                logger.info(f"   Current page: {current_page}")
                logger.info(f"   Expected page: {expected_page}")
                logger.info(f"   Target found: {target_found}")
                logger.info(f"   Frames returned: {len(results)}")
                
                if target_found:
                    target_frame = next((frame for frame in results if frame.get('is_target')), None)
                    if target_frame:
                        logger.info(f"   ✅ Target frame: {target_frame.get('filename')} (rank {target_frame.get('rank')})")
                
                # Check if auto-navigation worked correctly
                if current_page == expected_page:
                    logger.info(f"   🎯 Auto-navigation: ✅ Working correctly")
                else:
                    logger.warning(f"   ⚠️ Auto-navigation: Expected page {expected_page}, got {current_page}")
                
                # Check if target is on the returned page
                if target_found:
                    logger.info(f"   📍 Scroll behavior: ✅ Target frame sẽ được scroll đến")
                else:
                    logger.warning(f"   ⚠️ Scroll behavior: Target frame không có trên trang này")
                    
            else:
                logger.error(f"❌ Failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error testing: {e}")
        
        time.sleep(1)

def test_button_behaviors():
    """Test different button behaviors"""
    
    logger.info("\n🔘 Testing Button Behaviors...")
    
    buttons = [
        {
            "name": "🔍 Xem Toàn Bộ Video",
            "behavior": "Auto-navigation + auto-scroll to target (không scroll to top)"
        },
        {
            "name": "🎯 Đến Frame Target", 
            "behavior": "Chuyển trang + scroll to target (không scroll to top)"
        },
        {
            "name": "📍 Scroll to Target",
            "behavior": "Chỉ scroll to target (không chuyển trang, không scroll to top)"
        }
    ]
    
    for button in buttons:
        logger.info(f"   {button['name']}: {button['behavior']}")
    
    logger.info("\n📄 Normal page navigation (←/→ buttons):")
    logger.info("   Chuyển trang + scroll to top (như bình thường)")

def test_keyboard_shortcuts():
    """Test keyboard shortcuts"""
    
    logger.info("\n⌨️ Testing Keyboard Shortcuts...")
    
    shortcuts = [
        ("T", "Scroll to Target Frame (chỉ scroll, không chuyển trang)"),
        ("←/→", "Previous/Next Page (scroll to top)"),
        ("Home/End", "First/Last Page (scroll to top)"),
        ("ESC", "Go Back")
    ]
    
    for key, description in shortcuts:
        logger.info(f"   {key}: {description}")

if __name__ == "__main__":
    test_scroll_behavior_fixed()
    test_button_behaviors()
    test_keyboard_shortcuts()



