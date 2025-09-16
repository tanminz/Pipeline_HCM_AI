#!/usr/bin/env python3
"""
Search Strategies Module - Implement different search strategies for AI Challenge
Based on Lameframes reference implementation
"""

import json
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class SearchStrategies:
    def __init__(self, metadata_file="image_metadata.json"):
        self.metadata_file = metadata_file
        self.image_metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load image metadata"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.image_metadata = json.load(f)
            logger.info(f"Loaded {len(self.image_metadata)} images metadata")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.image_metadata = {}
    
    def single_search(self, query: str, k: int = 50) -> List[Dict]:
        """Single Search - Basic search strategy"""
        logger.info(f"Single Search: '{query}' with k={k}")
        
        # Simple mock search - in real implementation, this would use CLIP
        results = []
        query_lower = query.lower()
        
        for image_id, data in self.image_metadata.items():
            # Mock similarity based on query matching
            similarity = 0.0
            if query_lower in image_id.lower():
                similarity = 0.9
            elif query_lower in data['filename'].lower():
                similarity = 0.8
            else:
                # Random similarity for demo
                import random
                similarity = random.uniform(0.1, 0.7)
            
            if similarity > 0.1:  # Filter low similarity results
                results.append({
                    'id': data['id'],
                    'filename': data['filename'],
                    'path': data['original_path'],
                    'similarity': similarity,
                    'rank': len(results) + 1,
                    'video_source': data['filename'].split('/')[0]
                })
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def group_search(self, query: str, k: int = 50) -> Dict[str, List[Dict]]:
        """Group Search - Group results by video source"""
        logger.info(f"Group Search: '{query}' with k={k}")
        
        # Get single search results
        single_results = self.single_search(query, k * 2)  # Get more results for grouping
        
        # Group by video source
        grouped_results = defaultdict(list)
        for result in single_results:
            video_source = result['video_source']
            grouped_results[video_source].append(result)
        
        # Limit results per group
        final_groups = {}
        for video_source, results in grouped_results.items():
            final_groups[video_source] = results[:min(k // 2, len(results))]
        
        return dict(final_groups)
    
    def hierarchical_search(self, query: str, k1: int = 20, k2: int = 10) -> Dict[str, List[Dict]]:
        """Hierarchical Search - Combine Single and Group Search"""
        logger.info(f"Hierarchical Search: '{query}' with k1={k1}, k2={k2}")
        
        # Step 1: Single Search to get K1 initial results
        initial_results = self.single_search(query, k1)
        
        # Step 2: Group by video and get K2 results per video
        video_groups = defaultdict(list)
        for result in initial_results:
            video_source = result['video_source']
            video_groups[video_source].append(result)
        
        # Step 3: For each video, get K2 results
        final_results = {}
        for video_source, results in video_groups.items():
            # Get more results for this specific video
            video_results = self.local_search(query, video_source, k2)
            final_results[video_source] = video_results[:k2]
        
        return dict(final_results)
    
    def local_search(self, query: str, video_source: str, k: int = 20) -> List[Dict]:
        """Local Search - Search within a specific video"""
        logger.info(f"Local Search: '{query}' in video '{video_source}' with k={k}")
        
        results = []
        query_lower = query.lower()
        
        for image_id, data in self.image_metadata.items():
            # Check if image belongs to the specified video
            if not image_id.startswith(video_source):
                continue
            
            # Calculate similarity
            similarity = 0.0
            if query_lower in image_id.lower():
                similarity = 0.9
            elif query_lower in data['filename'].lower():
                similarity = 0.8
            else:
                import random
                similarity = random.uniform(0.1, 0.7)
            
            if similarity > 0.1:
                results.append({
                    'id': data['id'],
                    'filename': data['filename'],
                    'path': data['original_path'],
                    'similarity': similarity,
                    'rank': len(results) + 1,
                    'video_source': video_source
                })
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def fusion_search(self, query: str, sub_queries: List[str], weights: List[float], k: int = 50) -> List[Dict]:
        """Fusion Search - Combine multiple sub-queries with weights"""
        logger.info(f"Fusion Search: '{query}' with {len(sub_queries)} sub-queries")
        
        if len(sub_queries) != len(weights):
            raise ValueError("Number of sub-queries must match number of weights")
        
        # Get results for each sub-query
        all_results = {}
        for i, sub_query in enumerate(sub_queries):
            sub_results = self.single_search(sub_query, k)
            weight = weights[i]
            
            for result in sub_results:
                image_id = result['filename']
                if image_id not in all_results:
                    all_results[image_id] = {
                        'id': result['id'],
                        'filename': result['filename'],
                        'path': result['path'],
                        'similarity': 0.0,
                        'video_source': result['video_source']
                    }
                
                # Weighted fusion
                all_results[image_id]['similarity'] += result['similarity'] * weight
        
        # Convert to list and sort
        results = list(all_results.values())
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Add ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results[:k]
    
    def similarity_image_search(self, reference_image_id: int, k: int = 20) -> List[Dict]:
        """Similarity Image Search - Find similar images to a reference"""
        logger.info(f"Similarity Image Search: reference image {reference_image_id} with k={k}")
        
        # Find reference image
        reference_data = None
        for image_id, data in self.image_metadata.items():
            if data['id'] == reference_image_id:
                reference_data = data
                break
        
        if not reference_data:
            return []
        
        # Mock similarity calculation - in real implementation, this would use CLIP features
        results = []
        for image_id, data in self.image_metadata.items():
            if data['id'] == reference_image_id:
                continue  # Skip reference image
            
            # Mock similarity based on folder structure
            import random
            if data['filename'].split('/')[0] == reference_data['filename'].split('/')[0]:
                # Same video - higher similarity
                similarity = random.uniform(0.6, 0.9)
            else:
                # Different video - lower similarity
                similarity = random.uniform(0.1, 0.5)
            
            results.append({
                'id': data['id'],
                'filename': data['filename'],
                'path': data['original_path'],
                'similarity': similarity,
                'rank': len(results) + 1,
                'video_source': data['filename'].split('/')[0]
            })
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def get_available_videos(self) -> List[str]:
        """Get list of available video sources"""
        videos = set()
        for image_id in self.image_metadata.keys():
            video_source = image_id.split('/')[0]
            videos.add(video_source)
        return sorted(list(videos))
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to suggest best search strategy"""
        query_lower = query.lower()
        
        analysis = {
            'query': query,
            'length': len(query),
            'suggested_strategy': 'single',
            'confidence': 0.5,
            'features': {}
        }
        
        # Check for video-specific queries
        available_videos = self.get_available_videos()
        for video in available_videos:
            if video.lower() in query_lower:
                analysis['suggested_strategy'] = 'local'
                analysis['target_video'] = video
                analysis['confidence'] = 0.8
                break
        
        # Check for complex queries (multiple keywords)
        keywords = query_lower.split()
        if len(keywords) > 2:
            analysis['suggested_strategy'] = 'fusion'
            analysis['confidence'] = 0.7
            analysis['features']['multiple_keywords'] = True
        
        # Check for specific object queries
        object_keywords = ['car', 'person', 'building', 'animal', 'food']
        for keyword in object_keywords:
            if keyword in query_lower:
                analysis['features']['object_query'] = True
                analysis['confidence'] = max(analysis['confidence'], 0.6)
                break
        
        return analysis

def main():
    """Test the search strategies"""
    strategies = SearchStrategies()
    
    # Test different search strategies
    print("=== Testing Search Strategies ===")
    
    # Single Search
    print("\n1. Single Search:")
    results = strategies.single_search("car", 5)
    for result in results:
        print(f"  {result['rank']}. {result['filename']} (similarity: {result['similarity']:.2f})")
    
    # Group Search
    print("\n2. Group Search:")
    grouped = strategies.group_search("car", 10)
    for video, results in grouped.items():
        print(f"  Video {video}: {len(results)} results")
    
    # Local Search
    print("\n3. Local Search:")
    videos = strategies.get_available_videos()
    if videos:
        local_results = strategies.local_search("car", videos[0], 5)
        for result in local_results:
            print(f"  {result['rank']}. {result['filename']} (similarity: {result['similarity']:.2f})")
    
    # Query Analysis
    print("\n4. Query Analysis:")
    analysis = strategies.analyze_query("car driving in L01_V001")
    print(f"  Query: {analysis['query']}")
    print(f"  Suggested Strategy: {analysis['suggested_strategy']}")
    print(f"  Confidence: {analysis['confidence']:.2f}")

if __name__ == "__main__":
    main()




