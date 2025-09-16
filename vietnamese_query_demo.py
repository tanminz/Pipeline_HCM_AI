#!/usr/bin/env python3
"""
Vietnamese Query Processing Demo
Demo xử lý câu query tiếng Việt dài và phức tạp cho cuộc thi HCMC AI Challenge
"""

import requests
import json
import time
from datetime import datetime

def test_vietnamese_queries():
    """Test các câu query tiếng Việt phức tạp"""
    
    print("🇻🇳 VIETNAMESE QUERY PROCESSING DEMO")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Danh sách các câu query tiếng Việt phức tạp mô phỏng ban tổ chức
    vietnamese_queries = [
        # Textual KIS queries
        "Tìm những khung hình có xe ô tô màu đen đang di chuyển trên đường phố",
        "Hiển thị các ảnh có người đang đi bộ trên vỉa hè",
        "Tìm kiếm những frame có tòa nhà cao tầng và cây xanh",
        "Hiển thị các khung hình có biển quảng cáo và đèn đường",
        
        # Q&A queries
        "Có bao nhiêu người đang đứng trong khung hình này?",
        "Màu sắc của chiếc xe trong ảnh là gì?",
        "Có những loại phương tiện giao thông nào xuất hiện?",
        "Kiến trúc của tòa nhà trong ảnh thuộc loại gì?",
        
        # TRAKE queries (temporal reasoning)
        "Tìm chuỗi khung hình thể hiện quá trình một người từ đi bộ đến lên xe",
        "Hiển thị các frame thể hiện sự thay đổi ánh sáng từ ngày sang đêm",
        "Tìm kiếm chuỗi ảnh thể hiện sự di chuyển của một chiếc xe từ xa đến gần",
        "Hiển thị các khung hình thể hiện sự xuất hiện và biến mất của đám đông",
        
        # Complex queries
        "Tìm những ảnh có cả người đi bộ, xe máy và tòa nhà trong cùng một khung hình",
        "Hiển thị các frame có nhiều hơn 3 phương tiện giao thông khác nhau",
        "Tìm kiếm những khung hình có cả yếu tố tự nhiên (cây cối) và nhân tạo (nhà cửa)",
        "Hiển thị các ảnh có sự tương phản rõ rệt giữa ánh sáng và bóng tối"
    ]
    
    print(f"📝 Testing {len(vietnamese_queries)} Vietnamese queries...")
    print("-" * 60)
    
    results_summary = {
        "total_queries": len(vietnamese_queries),
        "successful_queries": 0,
        "failed_queries": 0,
        "average_response_time": 0,
        "query_results": []
    }
    
    response_times = []
    
    for i, query in enumerate(vietnamese_queries, 1):
        print(f"\n🔍 Query {i}/{len(vietnamese_queries)}:")
        print(f"   📝 Query: {query}")
        
        try:
            start_time = time.time()
            
            # Gửi query đến API
            response = requests.get(f"{base_url}/textsearch", params={
                'textquery': query,
                'k': 300,  # Lấy 300 kết quả
                'per_page': 300
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get('results', []))
                
                print(f"   ✅ Success! Found {results_count} results in {response_time:.2f}s")
                
                # Hiển thị một số kết quả mẫu
                if results_count > 0:
                    sample_results = data['results'][:3]  # Lấy 3 kết quả đầu
                    print(f"   📊 Sample results:")
                    for j, result in enumerate(sample_results, 1):
                        filename = result.get('filename', 'unknown')
                        similarity = result.get('similarity', 0.0)
                        print(f"      {j}. {filename} (similarity: {similarity:.3f})")
                
                results_summary["successful_queries"] += 1
                results_summary["query_results"].append({
                    "query": query,
                    "status": "success",
                    "results_count": results_count,
                    "response_time": response_time
                })
                
            else:
                print(f"   ❌ Failed with status code: {response.status_code}")
                results_summary["failed_queries"] += 1
                results_summary["query_results"].append({
                    "query": query,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results_summary["failed_queries"] += 1
            results_summary["query_results"].append({
                "query": query,
                "status": "failed",
                "error": str(e)
            })
    
    # Tính toán thống kê
    if response_times:
        results_summary["average_response_time"] = sum(response_times) / len(response_times)
    
    # In kết quả tổng kết
    print("\n" + "=" * 60)
    print("📊 QUERY PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total queries: {results_summary['total_queries']}")
    print(f"Successful: {results_summary['successful_queries']}")
    print(f"Failed: {results_summary['failed_queries']}")
    print(f"Success rate: {(results_summary['successful_queries']/results_summary['total_queries'])*100:.1f}%")
    print(f"Average response time: {results_summary['average_response_time']:.2f}s")
    
    # Đánh giá hiệu suất
    avg_time = results_summary["average_response_time"]
    if avg_time < 5:
        performance = "⚡ EXCELLENT"
    elif avg_time < 10:
        performance = "⚡ GOOD"
    elif avg_time < 15:
        performance = "⚡ ACCEPTABLE"
    else:
        performance = "⚡ SLOW"
    
    print(f"Performance: {performance}")
    
    return results_summary

def test_csv_export_with_vietnamese_queries():
    """Test xuất CSV với kết quả từ câu query tiếng Việt"""
    
    print("\n📋 CSV EXPORT WITH VIETNAMESE QUERIES")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test query tiếng Việt
    test_query = "Tìm những khung hình có xe ô tô màu đen đang di chuyển trên đường phố"
    
    try:
        print(f"🔍 Testing query: {test_query}")
        
        # Tìm kiếm
        response = requests.get(f"{base_url}/textsearch", params={
            'textquery': test_query,
            'k': 100,
            'per_page': 100
        })
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ Found {len(results)} results")
            
            # Xuất CSV cho từng loại nhiệm vụ
            csv_tasks = [
                {"type": "textual_kis", "name": "vietnamese_textual_kis"},
                {"type": "qa", "name": "vietnamese_qa"},
                {"type": "trake", "name": "vietnamese_trake"}
            ]
            
            for task in csv_tasks:
                try:
                    export_data = {
                        "task_type": task["type"],
                        "results": results,
                        "filename": task["name"],
                        "max_results": 100
                    }
                    
                    csv_response = requests.post(f"{base_url}/api/export_csv", json=export_data)
                    
                    if csv_response.status_code == 200:
                        csv_data = csv_response.json()
                        if csv_data.get('success'):
                            print(f"   ✅ {task['type'].upper()} CSV exported: {csv_data.get('filename')}")
                        else:
                            print(f"   ❌ {task['type'].upper()} CSV failed: {csv_data.get('error')}")
                    else:
                        print(f"   ❌ {task['type'].upper()} CSV HTTP error: {csv_response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ {task['type'].upper()} CSV exception: {e}")
        else:
            print(f"❌ Search failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def demonstrate_query_optimization():
    """Demo tối ưu hóa câu query tiếng Việt"""
    
    print("\n🚀 VIETNAMESE QUERY OPTIMIZATION TECHNIQUES")
    print("=" * 60)
    
    optimization_examples = [
        {
            "original": "Tìm những khung hình có xe ô tô màu đen đang di chuyển trên đường phố",
            "optimized": "xe ô tô đen đường phố",
            "technique": "Keyword extraction"
        },
        {
            "original": "Hiển thị các ảnh có người đang đi bộ trên vỉa hè",
            "optimized": "người đi bộ vỉa hè",
            "technique": "Remove stop words"
        },
        {
            "original": "Tìm kiếm những frame có tòa nhà cao tầng và cây xanh",
            "optimized": "tòa nhà cao tầng cây xanh",
            "technique": "Focus on key objects"
        },
        {
            "original": "Có bao nhiêu người đang đứng trong khung hình này?",
            "optimized": "người đứng",
            "technique": "Q&A simplification"
        }
    ]
    
    for i, example in enumerate(optimization_examples, 1):
        print(f"\n{i}. {example['technique']}:")
        print(f"   Original: {example['original']}")
        print(f"   Optimized: {example['optimized']}")

if __name__ == "__main__":
    print("🇻🇳 VIETNAMESE QUERY PROCESSING FOR HCMC AI CHALLENGE")
    print("=" * 80)
    
    # Test các câu query tiếng Việt
    results = test_vietnamese_queries()
    
    # Test CSV export
    test_csv_export_with_vietnamese_queries()
    
    # Demo tối ưu hóa
    demonstrate_query_optimization()
    
    print("\n🎉 Demo completed!")
    print("\n💡 Key Features for Vietnamese Queries:")
    print("   ✅ Natural language processing for Vietnamese")
    print("   ✅ Complex query understanding")
    print("   ✅ Object detection + Place recognition")
    print("   ✅ Logical relationships analysis")
    print("   ✅ CSV export for competition submission")
    print("   ✅ Performance optimization for long queries")



