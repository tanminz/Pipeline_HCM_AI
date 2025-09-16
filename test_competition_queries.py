#!/usr/bin/env python3
"""
Competition Query Test Script for HCMC AI Challenge V
Tests complex, long Vietnamese queries similar to competition scenarios
"""

import json
import time
import logging
from typing import List, Dict, Any
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompetitionQueryTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def test_query(self, query: str, expected_keywords: List[str] = None, description: str = ""):
        """Test a single complex query and analyze results"""
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
            translated_query = data.get('translated_query', None)
            
            logger.info(f"   ⏱️ Search time: {search_time:.2f}s")
            logger.info(f"   📊 Found: {count} results")
            logger.info(f"   🤖 AI Engine: {ai_engine}")
            logger.info(f"   🎯 Strategy: {strategy}")
            if translated_query:
                logger.info(f"   🌐 Translated: '{translated_query}'")
            
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
                'translated_query': translated_query,
                'top_results': results[:5] if results else [],
                'expected_keywords': expected_keywords,
                'found_keywords': found_keywords if expected_keywords else []
            }
            
            self.test_results.append(test_result)
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False
    
    def run_competition_tests(self):
        """Run competition-style complex query tests"""
        logger.info("🚀 Starting Competition Query Tests")
        logger.info("=" * 80)
        
        # Test 1: Animal + Location + Action queries
        logger.info("\n🐾 ANIMAL + LOCATION + ACTION QUERIES")
        self.test_query(
            "2 con trâu đi qua bãi cỏ xanh có một cái cây", 
            ["trâu", "cỏ", "cây"], 
            "Mô tả phức tạp về động vật và môi trường"
        )
        self.test_query(
            "3 con bò đang ăn cỏ trên đồng cỏ xanh", 
            ["bò", "cỏ"], 
            "Nhiều động vật với hành động cụ thể"
        )
        self.test_query(
            "con chó đen chạy trên đường phố", 
            ["chó"], 
            "Động vật với màu sắc và hành động"
        )
        self.test_query(
            "con mèo trắng ngồi trên ghế sofa", 
            ["mèo"], 
            "Động vật với màu sắc và vị trí"
        )
        
        # Test 2: People + Activity + Location queries
        logger.info("\n👥 PEOPLE + ACTIVITY + LOCATION QUERIES")
        self.test_query(
            "3 người đàn ông trượt tuyết trên núi", 
            ["người", "núi"], 
            "Nhiều người với hoạt động thể thao"
        )
        self.test_query(
            "2 phụ nữ đang nấu ăn trong nhà bếp", 
            ["phụ nữ", "nhà"], 
            "Nhiều người với hoạt động nội trợ"
        )
        self.test_query(
            "5 trẻ em chơi đùa trong công viên", 
            ["trẻ em"], 
            "Nhiều trẻ em với hoạt động vui chơi"
        )
        self.test_query(
            "người đàn ông lái xe máy trên đường", 
            ["người", "xe máy"], 
            "Người với phương tiện và hành động"
        )
        
        # Test 3: Mythical + Color + Action queries
        logger.info("\n🐉 MYTHICAL + COLOR + ACTION QUERIES")
        self.test_query(
            "con lân vàng nhảy múa", 
            ["lân"], 
            "Sinh vật thần thoại với màu sắc và hành động"
        )
        self.test_query(
            "con rồng đỏ bay trên bầu trời", 
            ["rồng"], 
            "Sinh vật thần thoại với màu sắc và môi trường"
        )
        self.test_query(
            "con phượng hoàng xanh đậu trên cây", 
            ["phượng hoàng"], 
            "Sinh vật thần thoại với màu sắc và vị trí"
        )
        
        # Test 4: Mathematical + Text queries
        logger.info("\n🔢 MATHEMATICAL + TEXT QUERIES")
        self.test_query(
            "Bài toán có đáp án là 51", 
            ["toán"], 
            "Nội dung toán học với số cụ thể"
        )
        self.test_query(
            "Phương trình x + y = 25", 
            ["phương trình"], 
            "Phương trình toán học"
        )
        self.test_query(
            "Bảng cửu chương nhân 7", 
            ["bảng", "nhân"], 
            "Bảng cửu chương"
        )
        
        # Test 5: Complex Scene + Object queries
        logger.info("\n🏞️ COMPLEX SCENE + OBJECT QUERIES")
        self.test_query(
            "xe hơi đỏ đậu trước tòa nhà cao tầng", 
            ["xe hơi", "tòa nhà"], 
            "Phương tiện với màu sắc và kiến trúc"
        )
        self.test_query(
            "máy bay trắng bay trên biển xanh", 
            ["máy bay", "biển"], 
            "Phương tiện với màu sắc và môi trường"
        )
        self.test_query(
            "tàu thủy lớn neo đậu ở cảng", 
            ["tàu", "cảng"], 
            "Phương tiện với kích thước và vị trí"
        )
        
        # Test 6: Weather + Time + Scene queries
        logger.info("\n🌤️ WEATHER + TIME + SCENE QUERIES")
        self.test_query(
            "trời mưa to trên đường phố ban đêm", 
            ["mưa", "đường"], 
            "Thời tiết với thời gian và địa điểm"
        )
        self.test_query(
            "nắng vàng chiếu sáng trên đồng lúa", 
            ["nắng", "lúa"], 
            "Thời tiết với màu sắc và môi trường"
        )
        self.test_query(
            "sương mù dày đặc bao phủ rừng", 
            ["sương mù", "rừng"], 
            "Thời tiết với môi trường"
        )
        
        # Test 7: Food + Cooking + Kitchen queries
        logger.info("\n🍳 FOOD + COOKING + KITCHEN QUERIES")
        self.test_query(
            "nồi cơm điện đang nấu cơm trắng", 
            ["nồi", "cơm"], 
            "Thiết bị nhà bếp với thực phẩm"
        )
        self.test_query(
            "bàn ăn có nhiều món ăn ngon", 
            ["bàn", "ăn"], 
            "Đồ nội thất với thực phẩm"
        )
        self.test_query(
            "tủ lạnh chứa đầy rau xanh", 
            ["tủ lạnh", "rau"], 
            "Thiết bị với thực phẩm"
        )
        
        # Test 8: Sports + Equipment + Location queries
        logger.info("\n⚽ SPORTS + EQUIPMENT + LOCATION QUERIES")
        self.test_query(
            "quả bóng đá đang lăn trên sân cỏ", 
            ["bóng", "sân"], 
            "Thể thao với thiết bị và địa điểm"
        )
        self.test_query(
            "vợt tennis và bóng vàng", 
            ["vợt", "bóng"], 
            "Thể thao với thiết bị và màu sắc"
        )
        self.test_query(
            "xe đạp đua chạy trên đường đua", 
            ["xe đạp", "đường"], 
            "Thể thao với phương tiện và địa điểm"
        )
        
        # Test 9: Technology + Office + Work queries
        logger.info("\n💻 TECHNOLOGY + OFFICE + WORK QUERIES")
        self.test_query(
            "máy tính để bàn có màn hình lớn", 
            ["máy tính", "màn hình"], 
            "Công nghệ với thiết bị"
        )
        self.test_query(
            "điện thoại di động đang sạc pin", 
            ["điện thoại"], 
            "Công nghệ với hành động"
        )
        self.test_query(
            "bàn làm việc có laptop và sách", 
            ["bàn", "laptop"], 
            "Văn phòng với thiết bị"
        )
        
        # Test 10: Nature + Season + Color queries
        logger.info("\n🌿 NATURE + SEASON + COLOR QUERIES")
        self.test_query(
            "hoa đào hồng nở vào mùa xuân", 
            ["hoa", "mùa xuân"], 
            "Thiên nhiên với mùa và màu sắc"
        )
        self.test_query(
            "lá cây vàng rơi mùa thu", 
            ["lá", "mùa thu"], 
            "Thiên nhiên với mùa và màu sắc"
        )
        self.test_query(
            "tuyết trắng phủ kín núi mùa đông", 
            ["tuyết", "núi"], 
            "Thiên nhiên với mùa và màu sắc"
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ All competition tests completed!")
        
    def generate_competition_report(self):
        """Generate a comprehensive competition test report"""
        logger.info("\n📊 GENERATING COMPETITION TEST REPORT")
        logger.info("=" * 80)
        
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
        
        with open('competition_query_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info("📄 Detailed report saved to: competition_query_report.json")
        
        # Show best and worst performing queries
        logger.info("\n🏆 BEST PERFORMING QUERIES:")
        best_queries = sorted(self.test_results, key=lambda x: x['count'], reverse=True)[:5]
        for i, result in enumerate(best_queries, 1):
            logger.info(f"   {i}. '{result['query']}' - {result['count']} results")
            
        logger.info("\n❌ WORST PERFORMING QUERIES:")
        worst_queries = sorted(self.test_results, key=lambda x: x['count'])[:5]
        for i, result in enumerate(worst_queries, 1):
            logger.info(f"   {i}. '{result['query']}' - {result['count']} results")
            
        # Show translation statistics
        translated_queries = [r for r in self.test_results if r['translated_query']]
        logger.info(f"\n🌐 Translation Statistics:")
        logger.info(f"   Queries translated: {len(translated_queries)}/{total_tests}")
        logger.info(f"   Translation rate: {len(translated_queries)/total_tests*100:.1f}%")

def main():
    """Main function"""
    tester = CompetitionQueryTester()
    
    try:
        # Test if server is running
        response = requests.get("http://localhost:5000/")
        if response.status_code != 200:
            logger.error("❌ Server not running! Please start the Flask app first.")
            return
            
        logger.info("✅ Server is running, starting competition tests...")
        
        # Run competition tests
        tester.run_competition_tests()
        
        # Generate report
        tester.generate_competition_report()
        
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to server! Please start the Flask app first.")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()


taa

