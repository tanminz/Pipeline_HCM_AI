#!/usr/bin/env python3
"""
Complex Query Test Script for HCMC AI Challenge V
Tests various complex and specific queries to evaluate search effectiveness
"""

import json
import time
import logging
from typing import List, Dict, Any
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplexQueryTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def test_query(self, query: str, expected_keywords: List[str] = None, description: str = ""):
        """Test a single query and analyze results"""
        logger.info(f"🔍 Testing: {query}")
        if description:
            logger.info(f"   Description: {description}")
            
        start_time = time.time()
        
        try:
            # Make API request
            response = requests.get(f"{self.base_url}/api/search", params={
                'q': query,
                'type': 'single',
                'k': 20
            })
            
            if response.status_code != 200:
                logger.error(f"❌ API Error: {response.status_code}")
                return False
                
            data = response.json()
            search_time = time.time() - start_time
            
            results = data.get('results', [])
            count = data.get('count', 0)
            strategy = data.get('strategy', 'unknown')
            ai_engine = data.get('ai_engine', False)
            
            logger.info(f"   ⏱️ Search time: {search_time:.2f}s")
            logger.info(f"   📊 Found: {count} results")
            logger.info(f"   🤖 AI Engine: {ai_engine}")
            logger.info(f"   🎯 Strategy: {strategy}")
            
            # Analyze top results
            if results:
                top_results = results[:5]
                logger.info("   🏆 Top 5 results:")
                
                for i, result in enumerate(top_results, 1):
                    filename = result.get('filename', 'unknown')
                    similarity = result.get('similarity', 0)
                    logger.info(f"      {i}. {filename} (similarity: {similarity:.3f})")
                    
                # Check if expected keywords appear in results
                if expected_keywords:
                    found_keywords = []
                    for keyword in expected_keywords:
                        for result in results:
                            filename = result.get('filename', '').lower()
                            if keyword.lower() in filename:
                                found_keywords.append(keyword)
                                break
                    
                    logger.info(f"   ✅ Found keywords: {found_keywords}")
                    logger.info(f"   ❌ Missing keywords: {[k for k in expected_keywords if k not in found_keywords]}")
            
            # Store test result
            test_result = {
                'query': query,
                'description': description,
                'search_time': search_time,
                'count': count,
                'strategy': strategy,
                'ai_engine': ai_engine,
                'top_results': results[:5] if results else [],
                'expected_keywords': expected_keywords,
                'found_keywords': found_keywords if expected_keywords else []
            }
            
            self.test_results.append(test_result)
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run comprehensive query tests"""
        logger.info("🚀 Starting Comprehensive Query Tests")
        logger.info("=" * 60)
        
        # Test 1: Animal queries
        logger.info("\n🐾 ANIMAL QUERIES")
        self.test_query("buffalo", ["buffalo", "animal"], "Tìm trâu/bò")
        self.test_query("con trâu", ["buffalo", "animal"], "Tìm trâu bằng tiếng Việt")
        self.test_query("cow", ["cow", "animal"], "Tìm bò")
        self.test_query("dog", ["dog", "animal"], "Tìm chó")
        self.test_query("cat", ["cat", "animal"], "Tìm mèo")
        self.test_query("bird", ["bird", "animal"], "Tìm chim")
        
        # Test 2: Vehicle queries
        logger.info("\n🚗 VEHICLE QUERIES")
        self.test_query("car", ["car", "vehicle"], "Tìm xe hơi")
        self.test_query("xe hơi", ["car", "vehicle"], "Tìm xe hơi bằng tiếng Việt")
        self.test_query("motorcycle", ["motorcycle", "bike"], "Tìm xe máy")
        self.test_query("bicycle", ["bicycle", "bike"], "Tìm xe đạp")
        self.test_query("truck", ["truck", "vehicle"], "Tìm xe tải")
        
        # Test 3: Person queries
        logger.info("\n👤 PERSON QUERIES")
        self.test_query("person", ["person", "people"], "Tìm người")
        self.test_query("người", ["person", "people"], "Tìm người bằng tiếng Việt")
        self.test_query("man", ["man", "person"], "Tìm đàn ông")
        self.test_query("woman", ["woman", "person"], "Tìm phụ nữ")
        self.test_query("child", ["child", "kid"], "Tìm trẻ em")
        
        # Test 4: Nature queries
        logger.info("\n🌿 NATURE QUERIES")
        self.test_query("tree", ["tree", "nature"], "Tìm cây")
        self.test_query("cây", ["tree", "nature"], "Tìm cây bằng tiếng Việt")
        self.test_query("flower", ["flower", "nature"], "Tìm hoa")
        self.test_query("mountain", ["mountain", "nature"], "Tìm núi")
        self.test_query("water", ["water", "river", "lake"], "Tìm nước/sông/hồ")
        
        # Test 5: Building queries
        logger.info("\n🏢 BUILDING QUERIES")
        self.test_query("building", ["building", "house"], "Tìm tòa nhà")
        self.test_query("nhà", ["house", "building"], "Tìm nhà bằng tiếng Việt")
        self.test_query("house", ["house", "building"], "Tìm nhà")
        self.test_query("bridge", ["bridge"], "Tìm cầu")
        
        # Test 6: Complex queries
        logger.info("\n🔍 COMPLEX QUERIES")
        self.test_query("person riding motorcycle", ["person", "motorcycle"], "Người đi xe máy")
        self.test_query("car on road", ["car", "road"], "Xe hơi trên đường")
        self.test_query("animal in nature", ["animal", "nature"], "Động vật trong thiên nhiên")
        self.test_query("building with people", ["building", "person"], "Tòa nhà có người")
        
        # Test 7: Specific object queries
        logger.info("\n📱 OBJECT QUERIES")
        self.test_query("phone", ["phone", "mobile"], "Tìm điện thoại")
        self.test_query("computer", ["computer", "laptop"], "Tìm máy tính")
        self.test_query("book", ["book"], "Tìm sách")
        self.test_query("chair", ["chair", "furniture"], "Tìm ghế")
        self.test_query("table", ["table", "furniture"], "Tìm bàn")
        
        # Test 8: Color queries
        logger.info("\n🎨 COLOR QUERIES")
        self.test_query("red car", ["car"], "Xe hơi màu đỏ")
        self.test_query("blue sky", ["sky"], "Bầu trời xanh")
        self.test_query("green tree", ["tree"], "Cây xanh")
        self.test_query("white building", ["building"], "Tòa nhà trắng")
        
        # Test 9: Action queries
        logger.info("\n🏃 ACTION QUERIES")
        self.test_query("running", ["person"], "Chạy")
        self.test_query("walking", ["person"], "Đi bộ")
        self.test_query("sitting", ["person"], "Ngồi")
        self.test_query("standing", ["person"], "Đứng")
        
        # Test 10: Scene queries
        logger.info("\n🏞️ SCENE QUERIES")
        self.test_query("city street", ["street", "city"], "Đường phố thành phố")
        self.test_query("rural area", ["rural", "countryside"], "Khu vực nông thôn")
        self.test_query("beach", ["beach", "sea"], "Bãi biển")
        self.test_query("forest", ["forest", "tree"], "Rừng")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed!")
        
    def generate_report(self):
        """Generate a comprehensive test report"""
        logger.info("\n📊 GENERATING TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['count'] > 0)
        avg_search_time = sum(r['search_time'] for r in self.test_results) / total_tests
        
        logger.info(f"📈 Total tests: {total_tests}")
        logger.info(f"✅ Successful searches: {successful_tests}")
        logger.info(f"❌ Failed searches: {total_tests - successful_tests}")
        logger.info(f"📊 Success rate: {successful_tests/total_tests*100:.1f}%")
        logger.info(f"⏱️ Average search time: {avg_search_time:.2f}s")
        
        # Save detailed report
        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': successful_tests/total_tests*100,
                'avg_search_time': avg_search_time
            },
            'test_results': self.test_results
        }
        
        with open('query_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info("📄 Detailed report saved to: query_test_report.json")
        
        # Show best and worst performing queries
        logger.info("\n🏆 BEST PERFORMING QUERIES:")
        best_queries = sorted(self.test_results, key=lambda x: x['count'], reverse=True)[:5]
        for i, result in enumerate(best_queries, 1):
            logger.info(f"   {i}. '{result['query']}' - {result['count']} results")
            
        logger.info("\n❌ WORST PERFORMING QUERIES:")
        worst_queries = sorted(self.test_results, key=lambda x: x['count'])[:5]
        for i, result in enumerate(worst_queries, 1):
            logger.info(f"   {i}. '{result['query']}' - {result['count']} results")

def main():
    """Main function"""
    tester = ComplexQueryTester()
    
    try:
        # Test if server is running
        response = requests.get("http://localhost:5000/")
        if response.status_code != 200:
            logger.error("❌ Server not running! Please start the Flask app first.")
            return
            
        logger.info("✅ Server is running, starting tests...")
        
        # Run comprehensive tests
        tester.run_comprehensive_tests()
        
        # Generate report
        tester.generate_report()
        
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to server! Please start the Flask app first.")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()



