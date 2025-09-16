#!/usr/bin/env python3
"""
Test script for AI Search Engine
"""

import json
import logging
from ai_search_engine import AISearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_search():
    """Test the search engine"""
    try:
        logger.info("🚀 Initializing AI Search Engine...")
        
        # Initialize engine
        engine = AISearchEngine("fast_image_metadata.json")
        
        # Test queries
        test_queries = [
            "person",
            "car", 
            "building",
            "nature",
            "run",
            "trâu",  # Vietnamese
            "xe hơi",  # Vietnamese
            "nhà"  # Vietnamese
        ]
        
        for query in test_queries:
            logger.info(f"🔍 Testing query: '{query}'")
            results = engine.search_by_text(query, k=5)
            logger.info(f"Found {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:3]):
                    logger.info(f"  {i+1}. {result['filename']} (similarity: {result['similarity']:.3f})")
            else:
                logger.info("  No results found")
            
            print("-" * 50)
        
        logger.info("✅ Search test completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_search()

