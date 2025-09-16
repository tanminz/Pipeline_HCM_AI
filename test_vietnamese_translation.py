#!/usr/bin/env python3
"""
Test Vietnamese Translation System
"""

import requests
import json

def test_translation_api():
    """Test the translation API endpoint"""
    base_url = "http://localhost:5000"
    
    # Test cases based on user feedback
    test_queries = [
        "con chó",
        "xe hơi", 
        "phụ nữ đội nón lá",
        "con lân vàng máu",
        "máy bay",
        "tàu thủy",
        "con trâu",
        "con bò",
        "người đàn ông",
        "trẻ em",
        "cây cối",
        "hoa đẹp",
        "núi cao",
        "biển xanh",
        "thành phố",
        "nông thôn",
        "nhà cửa",
        "đường phố",
        "công viên",
        "chợ đông người"
    ]
    
    print("🧪 Testing Vietnamese Translation System")
    print("=" * 50)
    
    for query in test_queries:
        try:
            # Test translation endpoint
            response = requests.get(f"{base_url}/api/translate_test", params={'q': query})
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query: '{query}'")
                print(f"   Translated: '{data.get('translated_query', 'N/A')}'")
                print(f"   Suggestions: {data.get('suggestions', [])}")
                print(f"   Translator: {data.get('translator', 'N/A')}")
                
                # Show analysis if available
                if 'analysis' in data:
                    analysis = data['analysis']
                    if 'components' in analysis:
                        components = analysis['components']
                        print(f"   Components:")
                        for comp_type, items in components.items():
                            if items:
                                print(f"     {comp_type}: {items}")
                
                print()
            else:
                print(f"❌ Query: '{query}' - HTTP {response.status_code}")
                print()
                
        except Exception as e:
            print(f"❌ Query: '{query}' - Error: {e}")
            print()

def test_search_with_translation():
    """Test search with translation"""
    base_url = "http://localhost:5000"
    
    # Test specific cases mentioned by user
    test_queries = [
        "con chó",
        "xe hơi",
        "phụ nữ đội nón lá", 
        "con lân vàng máu"
    ]
    
    print("🔍 Testing Search with Translation")
    print("=" * 50)
    
    for query in test_queries:
        try:
            response = requests.get(f"{base_url}/api/search_with_translation", params={'q': query, 'k': 10})
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query: '{query}'")
                print(f"   Translated: '{data.get('translation_info', {}).get('translated', 'N/A')}'")
                print(f"   Results: {data.get('count', 0)} found")
                
                # Show first few results
                results = data.get('results', [])
                if results:
                    print(f"   Top results:")
                    for i, result in enumerate(results[:3]):
                        print(f"     {i+1}. {result.get('filename', 'N/A')} (similarity: {result.get('similarity', 0):.3f})")
                
                print()
            else:
                print(f"❌ Query: '{query}' - HTTP {response.status_code}")
                print()
                
        except Exception as e:
            print(f"❌ Query: '{query}' - Error: {e}")
            print()

def test_enhanced_translator_directly():
    """Test the enhanced translator directly"""
    try:
        from enhanced_vietnamese_translator import translate_vietnamese_query, get_query_suggestions, analyze_vietnamese_query
        
        print("🔧 Testing Enhanced Translator Directly")
        print("=" * 50)
        
        test_queries = [
            "con chó",
            "xe hơi",
            "phụ nữ đội nón lá",
            "con lân vàng máu",
            "máy bay trắng",
            "tàu thủy lớn",
            "con trâu đen",
            "người đàn ông mặc áo",
            "cây cối xanh tươi",
            "hoa đẹp màu hồng"
        ]
        
        for query in test_queries:
            translated = translate_vietnamese_query(query)
            suggestions = get_query_suggestions(query)
            analysis = analyze_vietnamese_query(query)
            
            print(f"✅ Query: '{query}'")
            print(f"   Translated: '{translated}'")
            print(f"   Suggestions: {suggestions}")
            
            if analysis['components']:
                print(f"   Components:")
                for comp_type, items in analysis['components'].items():
                    if items:
                        print(f"     {comp_type}: {items}")
            
            print()
            
    except ImportError as e:
        print(f"❌ Enhanced translator not available: {e}")
    except Exception as e:
        print(f"❌ Error testing enhanced translator: {e}")

if __name__ == "__main__":
    print("🚀 Starting Vietnamese Translation Tests")
    print()
    
    # Test enhanced translator directly first
    test_enhanced_translator_directly()
    
    # Test API endpoints if server is running
    try:
        test_translation_api()
        test_search_with_translation()
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start the Flask app first with: python app.py")
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    print("✅ Translation tests completed!")



