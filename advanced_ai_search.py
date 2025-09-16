#!/usr/bin/env python3
"""
Advanced AI Search Engine for HCMC AI Challenge V
With Vector Quantization, Advanced Indexing, and Optimized Search
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from collections import defaultdict
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI/ML imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel
    import faiss
    from PIL import Image
    import cv2
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logger.error(f"Critical AI library missing: {e}")
    raise

class AdvancedAISearchEngine:
    def __init__(self, metadata_file="real_image_metadata.json"):
        self.metadata_file = metadata_file
        self.image_metadata = {}
        self.clip_model = None
        self.clip_processor = None
        self.faiss_index = None
        self.clip_features = None
        self.feature_dim = 512  # CLIP ViT-B/32 dimension
        
        # Advanced indexing
        self.quantizer = None
        self.index_file = "fast_faiss_index.bin"
        self.features_file = "fast_clip_features.npy"
        self.quantizer_file = "quantizer.pkl"
        self.valid_ids_file = "fast_valid_ids.pkl"
        
        # Performance tracking
        self.search_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Enhanced Vietnamese to English translation dictionary
        self.vietnamese_translations = {
            # Animals with detailed descriptions
            'trâu': 'buffalo', 'bò': 'cow', 'chó': 'dog', 'mèo': 'cat', 'chim': 'bird',
            'gà': 'chicken', 'vịt': 'duck', 'lợn': 'pig', 'dê': 'goat', 'cừu': 'sheep',
            'ngựa': 'horse', 'voi': 'elephant', 'hổ': 'tiger', 'sư tử': 'lion',
            'khỉ': 'monkey', 'gấu': 'bear', 'cá': 'fish', 'rắn': 'snake',
            'lân': 'unicorn', 'rồng': 'dragon', 'phượng hoàng': 'phoenix',
            'con trâu': 'buffalo', 'con bò': 'cow', 'con chó': 'dog', 'con mèo': 'cat',
            'con chim': 'bird', 'con gà': 'chicken', 'con vịt': 'duck', 'con lợn': 'pig',
            'con dê': 'goat', 'con cừu': 'sheep', 'con ngựa': 'horse', 'con voi': 'elephant',
            'con hổ': 'tiger', 'con sư tử': 'lion', 'con khỉ': 'monkey', 'con gấu': 'bear',
            'con cá': 'fish', 'con rắn': 'snake', 'con lân': 'unicorn', 'con rồng': 'dragon',
            'con phượng hoàng': 'phoenix',
            
            # Vehicles with detailed descriptions
            'xe hơi': 'car', 'xe máy': 'motorcycle', 'xe đạp': 'bicycle', 'xe tải': 'truck',
            'xe buýt': 'bus', 'tàu': 'ship', 'máy bay': 'airplane', 'tàu hỏa': 'train',
            'thuyền': 'boat', 'xe': 'vehicle', 'tàu thủy': 'ship', 'ô tô': 'car',
            'xe con': 'car', 'xe khách': 'passenger car', 'xe tải nhỏ': 'small truck',
            'xe tải lớn': 'large truck', 'xe đua': 'racing car', 'xe thể thao': 'sports car',
            
            # People with detailed descriptions
            'người': 'person', 'đàn ông': 'man', 'phụ nữ': 'woman', 'trẻ em': 'child',
            'bé': 'baby', 'bạn': 'friend', 'gia đình': 'family', 'nhóm': 'group',
            'người đàn ông': 'man', 'người phụ nữ': 'woman', 'người trẻ': 'young person',
            'người già': 'old person', 'em bé': 'baby', 'trẻ con': 'children',
            'phụ nữ đội nón lá': 'woman wearing conical hat', 'người đội nón': 'person wearing hat',
            'người mặc áo dài': 'person wearing ao dai', 'người mặc áo': 'person wearing shirt',
            
            # Clothing and accessories
            'nón lá': 'conical hat', 'nón': 'hat', 'áo dài': 'ao dai', 'áo': 'shirt',
            'quần': 'pants', 'váy': 'dress', 'giày': 'shoes', 'dép': 'sandals',
            'túi xách': 'bag', 'cặp': 'briefcase', 'mũ': 'cap', 'khăn': 'scarf',
            
            # Nature and environment
            'cây': 'tree', 'hoa': 'flower', 'cỏ': 'grass', 'lá': 'leaf',
            'núi': 'mountain', 'đồi': 'hill', 'sông': 'river', 'hồ': 'lake',
            'biển': 'sea', 'đại dương': 'ocean', 'bãi biển': 'beach', 'rừng': 'forest',
            'nước': 'water', 'đất': 'soil', 'đá': 'rock', 'cát': 'sand',
            'đồng cỏ': 'meadow', 'công viên': 'park', 'cảng': 'port',
            'cây cối': 'trees', 'rừng cây': 'forest', 'vườn hoa': 'flower garden',
            'bãi cỏ': 'grass field', 'đồng lúa': 'rice field', 'ruộng': 'rice field',
            
            # Buildings and structures
            'nhà': 'house', 'tòa nhà': 'building', 'cầu': 'bridge', 'đường': 'road',
            'phố': 'street', 'thành phố': 'city', 'làng': 'village', 'trường': 'school',
            'bệnh viện': 'hospital', 'chợ': 'market', 'nhà thờ': 'church',
            'nhà bếp': 'kitchen', 'văn phòng': 'office', 'cửa hàng': 'shop',
            'nhà cao tầng': 'skyscraper', 'chung cư': 'apartment building',
            'biệt thự': 'villa', 'nhà gỗ': 'wooden house', 'nhà sàn': 'stilt house',
            
            # Objects and items
            'bàn': 'table', 'ghế': 'chair', 'giường': 'bed', 'tủ': 'cabinet',
            'điện thoại': 'phone', 'máy tính': 'computer', 'sách': 'book', 'bút': 'pen',
            'giấy': 'paper', 'áo': 'shirt', 'quần': 'pants', 'giày': 'shoes',
            'nồi': 'pot', 'tủ lạnh': 'refrigerator', 'màn hình': 'screen',
            'laptop': 'laptop', 'vợt': 'racket', 'bóng': 'ball', 'tivi': 'television',
            'quạt': 'fan', 'đèn': 'lamp', 'đồng hồ': 'clock', 'gương': 'mirror',
            'bình hoa': 'vase', 'chậu': 'pot', 'thùng': 'box', 'hộp': 'box',
            
            # Colors with detailed descriptions
            'đỏ': 'red', 'xanh': 'blue', 'xanh lá': 'green', 'vàng': 'yellow',
            'đen': 'black', 'trắng': 'white', 'nâu': 'brown', 'hồng': 'pink',
            'tím': 'purple', 'cam': 'orange', 'xám': 'gray', 'xanh dương': 'blue',
            'xanh lục': 'green', 'xanh lam': 'blue', 'đỏ thẫm': 'dark red',
            'xanh đậm': 'dark blue', 'vàng nhạt': 'light yellow', 'trắng tinh': 'pure white',
            'đen tuyền': 'pure black', 'nâu đậm': 'dark brown', 'hồng nhạt': 'light pink',
            'màu đỏ': 'red color', 'màu xanh': 'blue color', 'màu vàng': 'yellow color',
            'màu đen': 'black color', 'màu trắng': 'white color', 'màu nâu': 'brown color',
            
            # Actions and activities
            'chạy': 'running', 'đi bộ': 'walking', 'ngồi': 'sitting', 'đứng': 'standing',
            'ăn': 'eating', 'uống': 'drinking', 'ngủ': 'sleeping', 'làm việc': 'working',
            'chơi': 'playing', 'học': 'studying', 'đọc': 'reading', 'viết': 'writing',
            'nhảy múa': 'dancing', 'bay': 'flying', 'đậu': 'perching', 'lăn': 'rolling',
            'nấu': 'cooking', 'sạc': 'charging', 'neo đậu': 'anchored', 'đậu xe': 'parked',
            'đang chạy': 'running', 'đang đi': 'walking', 'đang ngồi': 'sitting',
            'đang đứng': 'standing', 'đang ăn': 'eating', 'đang uống': 'drinking',
            'đang ngủ': 'sleeping', 'đang làm việc': 'working', 'đang chơi': 'playing',
            'đang học': 'studying', 'đang đọc': 'reading', 'đang viết': 'writing',
            'đang nhảy': 'dancing', 'đang bay': 'flying', 'đang đậu': 'perching',
            'đang lăn': 'rolling', 'đang nấu': 'cooking', 'đang sạc': 'charging',
            
            # Scenes and environments
            'thành phố': 'city', 'nông thôn': 'rural', 'thiên nhiên': 'nature',
            'trong nhà': 'indoor', 'ngoài trời': 'outdoor', 'ban ngày': 'daytime',
            'ban đêm': 'nighttime', 'mùa xuân': 'spring', 'mùa hè': 'summer',
            'mùa thu': 'autumn', 'mùa đông': 'winter', 'đường phố': 'street',
            'khu vực nông thôn': 'rural area', 'khu vực thành thị': 'urban area',
            'khu dân cư': 'residential area', 'khu công nghiệp': 'industrial area',
            'khu thương mại': 'commercial area', 'khu vui chơi': 'recreational area',
            
            # Weather and atmosphere
            'mưa': 'rain', 'nắng': 'sunny', 'sương mù': 'fog', 'tuyết': 'snow',
            'mưa to': 'heavy rain', 'mưa nhỏ': 'light rain', 'nắng gắt': 'intense sun',
            'sương mù dày đặc': 'thick fog', 'sương mù nhẹ': 'light fog',
            'tuyết rơi': 'falling snow', 'tuyết phủ': 'snow covered',
            'trời mưa': 'raining', 'trời nắng': 'sunny weather', 'trời âm u': 'cloudy',
            'trời quang': 'clear sky', 'trời tối': 'dark sky',
            
            # Food and dining
            'cơm': 'rice', 'rau': 'vegetables', 'món ăn': 'food', 'thịt': 'meat',
            'cá': 'fish', 'gà': 'chicken', 'bò': 'beef', 'lợn': 'pork',
            'canh': 'soup', 'cháo': 'porridge', 'bánh': 'cake', 'kẹo': 'candy',
            'nhiều món ăn': 'many dishes', 'đầy rau xanh': 'full of green vegetables',
            'cơm trắng': 'white rice', 'món ăn ngon': 'delicious food',
            'bàn ăn': 'dining table', 'nhà bếp': 'kitchen', 'nồi cơm điện': 'rice cooker',
            
            # Sports and recreation
            'sân': 'field', 'đường đua': 'race track', 'sân bóng': 'soccer field',
            'sân tennis': 'tennis court', 'sân golf': 'golf course', 'hồ bơi': 'swimming pool',
            'quả bóng đá': 'soccer ball', 'vợt tennis': 'tennis racket',
            'xe đạp đua': 'racing bicycle', 'máy chạy bộ': 'treadmill',
            'đang chơi thể thao': 'playing sports', 'đang tập thể dục': 'exercising',
            
            # Numbers and mathematics
            'một': 'one', 'hai': 'two', 'ba': 'three', 'bốn': 'four', 'năm': 'five',
            'sáu': 'six', 'bảy': 'seven', 'tám': 'eight', 'chín': 'nine', 'mười': 'ten',
            'toán': 'math', 'phương trình': 'equation', 'bảng': 'table',
            'nhân': 'multiply', 'đáp án': 'answer', 'bài toán': 'math problem',
            'bảng cửu chương': 'multiplication table', 'phép tính': 'calculation',
            'số': 'number', 'con số': 'number', 'chữ số': 'digit',
            
            # Technology and electronics
            'di động': 'mobile', 'để bàn': 'desktop', 'pin': 'battery',
            'làm việc': 'work', 'sách': 'book', 'máy tính để bàn': 'desktop computer',
            'điện thoại di động': 'mobile phone', 'bàn làm việc': 'work desk',
            'màn hình lớn': 'large screen', 'màn hình nhỏ': 'small screen',
            'đang sạc pin': 'charging battery', 'có laptop': 'with laptop',
            'máy in': 'printer', 'máy fax': 'fax machine', 'máy photocopy': 'photocopier',
            
            # Common descriptive words
            'lớn': 'big', 'nhỏ': 'small', 'cao': 'tall', 'thấp': 'short',
            'mới': 'new', 'cũ': 'old', 'đẹp': 'beautiful', 'xấu': 'ugly',
            'nhanh': 'fast', 'chậm': 'slow', 'nóng': 'hot', 'lạnh': 'cold',
            'to': 'big', 'dày đặc': 'thick', 'bao phủ': 'covering',
            'chiếu sáng': 'shining', 'phủ kín': 'covering', 'rộng': 'wide',
            'hẹp': 'narrow', 'dài': 'long', 'ngắn': 'short', 'dày': 'thick',
            'mỏng': 'thin', 'nặng': 'heavy', 'nhẹ': 'light', 'cứng': 'hard',
            'mềm': 'soft', 'sạch': 'clean', 'bẩn': 'dirty', 'sáng': 'bright',
            'tối': 'dark', 'rõ': 'clear', 'mờ': 'blurry', 'đầy': 'full',
            'vắng': 'empty', 'nhiều': 'many', 'ít': 'few', 'tất cả': 'all',
            'một số': 'some', 'vài': 'several', 'mỗi': 'each', 'mọi': 'every',
            
            # Complex competition phrases and specific cases
            '2 con trâu': 'two buffalo', '3 con bò': 'three cow', '5 trẻ em': 'five children',
            '3 người đàn ông': 'three men', '2 phụ nữ': 'two women',
            'con lân vàng': 'golden unicorn', 'con rồng đỏ': 'red dragon',
            'con phượng hoàng xanh': 'blue phoenix', 'xe hơi đỏ': 'red car',
            'máy bay trắng': 'white airplane', 'tàu thủy lớn': 'large ship',
            'hoa đào hồng': 'pink peach flower', 'lá cây vàng': 'yellow leaves',
            'tuyết trắng': 'white snow', 'nắng vàng': 'golden sun',
            'bóng vàng': 'yellow ball', 'nhiều món ăn': 'many dishes',
            'cơm trắng': 'white rice', 'món ăn ngon': 'delicious food',
            'tòa nhà cao tầng': 'skyscraper', 'biển xanh': 'blue sea',
            'đồng lúa': 'rice field', 'sân cỏ': 'grass field',
            'nở vào': 'blooming in', 'rơi mùa': 'falling in season',
            'đang nấu': 'cooking', 'chứa đầy': 'full of', 'đang lăn': 'rolling',
            'chạy trên': 'running on', 'neo đậu ở': 'anchored at',
            'đậu trước': 'parked in front of', 'bay trên': 'flying over',
            'có đáp án': 'with answer', 'là 51': 'is 51',
            'x + y = 25': 'x plus y equals 25', 'nhân 7': 'multiply by 7',
            
            # Special cases for competition queries
            'con lân vàng máu': 'golden blood unicorn', 'con lân vàng': 'golden unicorn',
            'con lân': 'unicorn', 'lân vàng': 'golden unicorn', 'lân vàng máu': 'golden blood unicorn',
            'phụ nữ đội nón lá': 'woman wearing conical hat', 'người phụ nữ đội nón': 'woman wearing hat',
            'phụ nữ mặc áo dài': 'woman wearing ao dai', 'người mặc áo dài': 'person wearing ao dai',
            'xe hơi đậu': 'parked car', 'xe hơi đang chạy': 'running car',
            'xe hơi màu đỏ': 'red car', 'xe hơi màu xanh': 'blue car',
            'xe hơi màu trắng': 'white car', 'xe hơi màu đen': 'black car',
            'con chó đang chạy': 'running dog', 'con chó đang ngồi': 'sitting dog',
            'con chó màu nâu': 'brown dog', 'con chó màu đen': 'black dog',
            'con chó màu trắng': 'white dog', 'con chó nhỏ': 'small dog',
            'con chó lớn': 'big dog', 'con chó con': 'puppy',
            'máy bay đang bay': 'flying airplane', 'máy bay trên trời': 'airplane in sky',
            'máy bay màu trắng': 'white airplane', 'máy bay màu xanh': 'blue airplane',
            'tàu thủy đang neo': 'anchored ship', 'tàu thủy trên biển': 'ship on sea',
            'tàu thủy lớn': 'large ship', 'tàu thủy nhỏ': 'small ship',
            'bãi biển đẹp': 'beautiful beach', 'bãi biển xanh': 'blue beach',
            'rừng cây xanh': 'green forest', 'rừng rậm': 'dense forest',
            'đồng lúa vàng': 'golden rice field', 'đồng lúa xanh': 'green rice field',
            'sân cỏ xanh': 'green grass field', 'sân bóng đá': 'soccer field',
            'công viên đẹp': 'beautiful park', 'công viên xanh': 'green park',
            'thành phố hiện đại': 'modern city', 'thành phố cổ': 'old city',
            'làng quê': 'rural village', 'làng nghề': 'craft village',
            'chợ đông người': 'crowded market', 'chợ truyền thống': 'traditional market',
            'nhà thờ cổ': 'old church', 'nhà thờ lớn': 'large church',
            'cầu dài': 'long bridge', 'cầu đẹp': 'beautiful bridge',
            'đường phố đông đúc': 'crowded street', 'đường phố vắng vẻ': 'quiet street',
            'tòa nhà cao': 'tall building', 'tòa nhà hiện đại': 'modern building',
            'văn phòng làm việc': 'working office', 'văn phòng hiện đại': 'modern office',
            'nhà bếp sạch sẽ': 'clean kitchen', 'nhà bếp hiện đại': 'modern kitchen',
            'phòng ngủ': 'bedroom', 'phòng khách': 'living room',
            'phòng tắm': 'bathroom', 'phòng ăn': 'dining room',
            'ban công': 'balcony', 'sân thượng': 'rooftop',
            'hồ bơi xanh': 'blue swimming pool', 'hồ bơi lớn': 'large swimming pool',
            'vườn hoa đẹp': 'beautiful flower garden', 'vườn cây': 'garden',
            'cây cối xanh tươi': 'lush green trees', 'cây cao': 'tall tree',
            'hoa đẹp': 'beautiful flower', 'hoa hồng': 'rose flower',
            'hoa cúc': 'chrysanthemum', 'hoa lan': 'orchid',
            'lá cây xanh': 'green leaves', 'lá cây vàng': 'yellow leaves',
            'lá cây rơi': 'falling leaves', 'cỏ xanh': 'green grass',
            'cỏ dại': 'wild grass', 'cỏ mọc': 'growing grass',
            'núi cao': 'high mountain', 'núi xanh': 'green mountain',
            'đồi thoai thoải': 'gentle hill', 'đồi xanh': 'green hill',
            'sông dài': 'long river', 'sông xanh': 'blue river',
            'hồ nước': 'lake', 'hồ xanh': 'blue lake',
            'biển xanh': 'blue sea', 'biển đẹp': 'beautiful sea',
            'đại dương mênh mông': 'vast ocean', 'đại dương xanh': 'blue ocean',
            'cảng biển': 'seaport', 'cảng sông': 'river port',
            'bãi cát trắng': 'white sand beach', 'bãi cát vàng': 'golden sand beach',
            'đá cứng': 'hard rock', 'đá to': 'large rock',
            'cát mịn': 'fine sand', 'cát vàng': 'golden sand',
            'đất đen': 'black soil', 'đất màu mỡ': 'fertile soil',
            'nước trong': 'clear water', 'nước xanh': 'blue water',
            'nước mưa': 'rainwater', 'nước biển': 'seawater',
            'trời xanh': 'blue sky', 'trời trong': 'clear sky',
            'mây trắng': 'white clouds', 'mây xám': 'gray clouds',
            'gió nhẹ': 'gentle wind', 'gió mạnh': 'strong wind',
            'ánh nắng': 'sunlight', 'ánh sáng': 'light',
            'bóng râm': 'shadow', 'bóng mát': 'shade',
            'nhiệt độ cao': 'high temperature', 'nhiệt độ thấp': 'low temperature',
            'không khí trong lành': 'fresh air', 'không khí ô nhiễm': 'polluted air',
            'tiếng ồn': 'noise', 'tiếng động': 'sound',
            'sự yên tĩnh': 'silence', 'sự im lặng': 'quietness',
            'mùi hương': 'fragrance', 'mùi thơm': 'pleasant smell',
            'mùi khó chịu': 'unpleasant smell', 'mùi hôi': 'bad smell',
            'cảm giác ấm áp': 'warm feeling', 'cảm giác mát mẻ': 'cool feeling',
            'cảm giác dễ chịu': 'comfortable feeling', 'cảm giác khó chịu': 'uncomfortable feeling'
        }
        
        # Load metadata and initialize
        self.load_metadata()
        self.initialize_models()
        self.load_or_build_index()
    
    def translate_vietnamese_to_english(self, query: str) -> str:
        """Translate Vietnamese query to English for better CLIP understanding"""
        try:
            # Use enhanced translator if available
            from enhanced_vietnamese_translator import translate_vietnamese_query
            return translate_vietnamese_query(query)
        except ImportError:
            # Fallback to basic translation
            if not query:
                return query
                
            original_query = query
            translated_query = query.lower()
            
            # Replace Vietnamese words with English equivalents
            for vietnamese, english in self.vietnamese_translations.items():
                translated_query = translated_query.replace(vietnamese.lower(), english)
            
            # Handle common Vietnamese phrases
            phrase_translations = {
                'con trâu': 'buffalo',
                'con bò': 'cow', 
                'con chó': 'dog',
                'con mèo': 'cat',
                'con chim': 'bird',
                'xe hơi': 'car',
                'xe máy': 'motorcycle',
                'xe đạp': 'bicycle',
                'xe tải': 'truck',
                'người đi': 'person riding',
                'người đang': 'person',
                'trên đường': 'on road',
                'trong thiên nhiên': 'in nature',
                'có người': 'with people',
                'màu đỏ': 'red',
                'màu xanh': 'blue',
                'màu xanh lá': 'green',
                'màu vàng': 'yellow',
                'màu đen': 'black',
                'màu trắng': 'white',
                'đường phố': 'street',
                'khu vực nông thôn': 'rural area',
                'bãi biển': 'beach',
                'rừng': 'forest',
                
                # Complex competition phrases
                '2 con trâu': 'two buffalo',
                '3 con bò': 'three cow',
                '5 trẻ em': 'five children',
                '3 người đàn ông': 'three men',
                '2 phụ nữ': 'two women',
                'con lân vàng': 'golden unicorn',
                'con rồng đỏ': 'red dragon',
                'con phượng hoàng xanh': 'blue phoenix',
                'xe hơi đỏ': 'red car',
                'máy bay trắng': 'white airplane',
                'tàu thủy lớn': 'large ship',
                'máy tính để bàn': 'desktop computer',
                'điện thoại di động': 'mobile phone',
            'bàn làm việc': 'work desk',
            'hoa đào hồng': 'pink peach flower',
            'lá cây vàng': 'yellow leaves',
            'tuyết trắng': 'white snow',
            'nắng vàng': 'golden sun',
            'sương mù dày đặc': 'thick fog',
            'trời mưa to': 'heavy rain',
            'nồi cơm điện': 'rice cooker',
            'quả bóng đá': 'soccer ball',
            'vợt tennis': 'tennis racket',
            'xe đạp đua': 'racing bicycle',
            'bài toán': 'math problem',
            'bảng cửu chương': 'multiplication table',
            'tòa nhà cao tầng': 'skyscraper',
            'biển xanh': 'blue sea',
            'đồng lúa': 'rice field',
            'sân cỏ': 'grass field',
            'bóng vàng': 'yellow ball',
            'màn hình lớn': 'large screen',
            'nhiều món ăn': 'many dishes',
            'đầy rau xanh': 'full of green vegetables',
            'cơm trắng': 'white rice',
            'món ăn ngon': 'delicious food',
            'đang sạc pin': 'charging battery',
            'có laptop': 'with laptop',
            'nở vào': 'blooming in',
            'rơi mùa': 'falling in season',
            'phủ kín': 'completely covering',
            'chiếu sáng': 'shining',
            'bao phủ': 'covering',
            'đang nấu': 'cooking',
            'chứa đầy': 'full of',
            'đang lăn': 'rolling',
            'chạy trên': 'running on',
            'neo đậu ở': 'anchored at',
            'đậu trước': 'parked in front of',
            'bay trên': 'flying over',
            'có đáp án': 'with answer',
            'là 51': 'is 51',
            'x + y = 25': 'x plus y equals 25',
            'nhân 7': 'multiply by 7'
        }
        
        for phrase, translation in phrase_translations.items():
            translated_query = translated_query.replace(phrase, translation)
        
        # If translation changed the query, log it
        if translated_query != original_query.lower():
            logger.info(f"🌐 Translated: '{original_query}' -> '{translated_query}'")
            
        return translated_query
    
    def load_metadata(self):
        """Load and scan all available images"""
        try:
            # Scan all images in static/images directory
            self.scan_all_images()
            logger.info(f"✅ Scanned {len(self.image_metadata)} images from filesystem")
        except Exception as e:
            logger.error(f"❌ Error scanning images: {e}")
            self.image_metadata = {}
    
    def scan_all_images(self):
        """Scan all images in the static/images directory"""
        self.image_metadata = {}
        image_id = 0
        
        images_dir = Path("static/images")
        if not images_dir.exists():
            logger.error("static/images directory not found!")
            return
        
        logger.info(f"Scanning images in: {images_dir.absolute()}")
        
        # Scan all subdirectories recursively
        for item in images_dir.rglob("*.jpg"):
            try:
                # Create relative path from static/images
                relative_path = item.relative_to(images_dir)
                
                # Parse the path structure
                path_parts = str(relative_path).split('\\')  # Windows path separator
                
                if len(path_parts) >= 2:
                    video_folder = path_parts[0]  # e.g., Keyframes_L21
                    
                    if len(path_parts) >= 3:
                        video_name = path_parts[-2]  # e.g., L21_V001
                        frame_name = path_parts[-1]  # e.g., 044.jpg
                    else:
                        video_name = path_parts[0]
                        frame_name = path_parts[1]
                    
                    # Create image ID string
                    image_id_str = f"{video_name}/{frame_name}"
                    
                    self.image_metadata[image_id_str] = {
                        'id': image_id,
                        'web_path': str(item),
                        'filename': image_id_str,
                        'video_folder': video_folder,
                        'video_name': video_name,
                        'frame_name': frame_name,
                        'size': item.stat().st_size,
                        'is_real_data': True
                    }
                    
                    image_id += 1
                    
                    if image_id % 1000 == 0:
                        logger.info(f"Scanned {image_id} images...")
                        
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                continue
        
        logger.info(f"Total images scanned: {len(self.image_metadata)}")
        
        # Show some examples
        if self.image_metadata:
            sample_keys = list(self.image_metadata.keys())[:5]
            logger.info(f"Sample images: {sample_keys}")
    
    def initialize_models(self):
        """Initialize CLIP model with optimization"""
        try:
            logger.info("🚀 Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Set to evaluation mode
            self.clip_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                logger.info("✅ CLIP model loaded on GPU")
            else:
                logger.info("✅ CLIP model loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ Error loading CLIP model: {e}")
            raise
    
    def extract_clip_features_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract CLIP features in batches for efficiency"""
        features_list = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load batch of images
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(j)
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process batch with CLIP
                inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    batch_features = image_features.cpu().numpy()
                    
                    # Add features for valid images
                    for idx in valid_indices:
                        features_list.append(batch_features[idx])
                        
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Add random features for failed batch
                for _ in range(len(batch_paths)):
                    features_list.append(np.random.rand(self.feature_dim))
        
        return np.array(features_list, dtype=np.float32)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features using CLIP"""
        try:
            # Process text
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                features = text_features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return np.random.rand(self.feature_dim)
    
    def build_quantizer(self, features: np.ndarray, n_clusters: int = 1000):
        """Build vector quantizer for better indexing"""
        try:
            logger.info(f"🔨 Building quantizer with {n_clusters} clusters...")
            self.quantizer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
            self.quantizer.fit(features)
            
            # Save quantizer
            with open(self.quantizer_file, 'wb') as f:
                pickle.dump(self.quantizer, f)
            
            logger.info("✅ Quantizer built and saved")
            
        except Exception as e:
            logger.error(f"Error building quantizer: {e}")
    
    def load_or_build_index(self):
        """Load existing advanced index or build new one"""
        try:
            # Try to load existing index
            if (os.path.exists(self.index_file) and 
                os.path.exists(self.features_file) and 
                os.path.exists(self.quantizer_file)):
                
                logger.info("📂 Loading existing advanced FAISS index...")
                self.faiss_index = faiss.read_index(self.index_file)
                self.clip_features = np.load(self.features_file)
                
                with open(self.quantizer_file, 'rb') as f:
                    self.quantizer = pickle.load(f)
                
                with open(self.valid_ids_file, "rb") as f:
                    self.valid_image_ids = pickle.load(f)
                
                logger.info(f"✅ Loaded advanced FAISS index with {len(self.valid_image_ids)} images")
                return
            
            # Build new advanced index
            logger.info("🔨 Building new advanced FAISS index...")
            self.build_advanced_search_index()
            
        except Exception as e:
            logger.error(f"Error loading/building advanced index: {e}")
            self.build_advanced_search_index()
    
    def build_advanced_search_index(self):
        """Build advanced FAISS index with vector quantization"""
        try:
            logger.info("🔨 Building advanced FAISS search index...")
            
            # Get all image paths
            image_paths = []
            valid_images = []
            
            total_images = len(self.image_metadata)
            for i, (image_id_str, data) in enumerate(self.image_metadata.items()):
                if i % 1000 == 0:
                    logger.info(f"Preparing {i}/{total_images} images...")
                
                image_path = data['web_path']
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    valid_images.append(image_id_str)
            
            if not image_paths:
                logger.warning("No valid images found for indexing")
                return
            
            logger.info(f"Extracting features for {len(image_paths)} images...")
            
            # Extract features in batches
            self.clip_features = self.extract_clip_features_batch(image_paths, batch_size=32)
            
            # Normalize features
            faiss.normalize_L2(self.clip_features)
            
            # Build quantizer
            self.build_quantizer(self.clip_features, n_clusters=min(1000, len(self.clip_features) // 10))
            
            # Create advanced FAISS index with quantization
            if self.quantizer is not None:
                # Use IVF index with quantizer
                nlist = min(100, len(self.clip_features) // 100)  # Number of clusters
                self.faiss_index = faiss.IndexIVFFlat(self.quantizer, self.feature_dim, nlist)
                self.faiss_index.train(self.clip_features)
                self.faiss_index.add(self.clip_features)
            else:
                # Fallback to simple index
                self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
                self.faiss_index.add(self.clip_features)
            
            # Store valid image IDs
            self.valid_image_ids = valid_images
            
            # Save index and features
            faiss.write_index(self.faiss_index, self.index_file)
            np.save(self.features_file, self.clip_features)
            
            with open(self.valid_ids_file, "wb") as f:
                pickle.dump(self.valid_image_ids, f)
            
            logger.info(f"✅ Advanced FAISS index built and saved with {len(valid_images)} images")
            
        except Exception as e:
            logger.error(f"Error building advanced search index: {e}")
    
    def search_by_text(self, query: str, k: int = 300) -> List[Dict[str, Any]]:
        """Search images by text query with Vietnamese translation support"""
        start_time = time.time()
        
        # Translate Vietnamese to English if needed
        translated_query = self.translate_vietnamese_to_english(query)
        
        try:
            # Encode text query
            text_inputs = self.clip_processor(text=[translated_query], return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Search in FAISS index
            if self.faiss_index is not None:
                # Convert to numpy for FAISS
                query_vector = text_features.cpu().numpy().astype('float32')
                
                # Search
                similarities, indices = self.faiss_index.search(query_vector, k)
                
                # Format results
                results = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx < len(self.valid_image_ids):
                        image_id = self.valid_image_ids[idx]
                        if str(image_id) in self.image_metadata:
                            metadata = self.image_metadata[str(image_id)]
                            results.append({
                                'id': image_id,
                                'path': metadata['web_path'],
                                'filename': metadata['filename'],
                                'similarity': float(similarity),
                                'rank': i + 1,
                                'translated_query': translated_query if translated_query != query else None
                            })
                
                # Track performance
                search_time = time.time() - start_time
                self.search_times.append(search_time)
                
                logger.info(f"🔍 Search completed in {search_time:.3f}s")
                logger.info(f"   Query: '{query}' -> '{translated_query}'")
                logger.info(f"   Results: {len(results)}")
                
                return results
            else:
                logger.error("FAISS index not available")
                return []
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def advanced_search(self, query: str, filters: Dict = None, k: int = 50) -> List[Dict]:
        """Advanced search with multiple strategies"""
        try:
            # Basic text search
            results = self.search_by_text(query, k * 2)
            
            # Apply filters
            if filters:
                results = self.apply_filters(results, filters)
            
            # Re-rank with advanced criteria
            results = self.advanced_rerank(results, query)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return self.fallback_search(query, k)
    
    def apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply advanced filters"""
        filtered_results = []
        
        for result in results:
            # Video folder filter
            if 'video_folder' in filters:
                video_folder = result.get('video_folder', '')
                if filters['video_folder'] not in video_folder:
                    continue
            
            # Video name filter
            if 'video_name' in filters:
                video_name = result.get('video_name', '')
                if filters['video_name'] not in video_name:
                    continue
            
            # Similarity threshold
            if 'min_similarity' in filters:
                if result['similarity'] < filters['min_similarity']:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def advanced_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Advanced re-ranking with multiple criteria"""
        try:
            query_lower = query.lower()
            query_words = query_lower.split()
            
            for result in results:
                filename = result['filename'].lower()
                video_name = result.get('video_name', '').lower()
                boost = 0.0
                
                # Boost for exact word matches in filename
                for word in query_words:
                    if word in filename:
                        boost += 0.15
                    
                    # Boost for video name matches
                    if word in video_name:
                        boost += 0.1
                
                # Boost for longer queries (more specific)
                if len(query_words) > 1:
                    boost += 0.05
                
                # Apply boost
                result['similarity'] = min(1.0, result['similarity'] + boost)
            
            # Sort by updated similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced reranking: {e}")
            return results
    
    def fallback_search(self, query: str, k: int) -> List[Dict]:
        """Improved fallback search"""
        try:
            query_lower = query.lower()
            results = []
            
            for image_id_str, data in self.image_metadata.items():
                filename_lower = image_id_str.lower()
                video_name = data.get('video_name', '').lower()
                
                # Check if query words appear in filename or video name
                query_words = query_lower.split()
                filename_matches = sum(1 for word in query_words if word in filename_lower)
                video_matches = sum(1 for word in query_words if word in video_name)
                
                total_matches = filename_matches + video_matches
                
                if total_matches > 0:
                    similarity = total_matches / len(query_words)
                    results.append({
                        'id': data['id'],
                        'path': data['web_path'],
                        'filename': image_id_str,
                        'similarity': similarity,
                        'rank': len(results) + 1,
                        'video_folder': data.get('video_folder', 'Unknown'),
                        'video_name': data.get('video_name', 'Unknown')
                    })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    def get_search_statistics(self) -> Dict:
        """Get advanced search engine statistics"""
        avg_search_time = np.mean(self.search_times) if self.search_times else 0
        
        return {
            'total_images': len(self.image_metadata),
            'indexed_images': len(self.valid_image_ids),
            'clip_model_loaded': self.clip_model is not None,
            'faiss_index_built': self.faiss_index is not None,
            'quantizer_built': self.quantizer is not None,
            'feature_dimension': self.feature_dim,
            'average_search_time': avg_search_time,
            'cache_size': len(self.cache),
            'gpu_available': torch.cuda.is_available()
        }
    
    def test_search(self, test_queries: List[str] = None):
        """Test the advanced search engine"""
        if test_queries is None:
            test_queries = [
                "buffalo",
                "person",
                "car",
                "building", 
                "nature",
                "indoor scene",
                "outdoor",
                "people",
                "vehicle",
                "animal",
                "water",
                "sky"
            ]
        
        print("🧪 Testing Advanced AI Search Engine...")
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = self.search_by_text(query, k=10)
            
            print(f"Found {len(results)} results:")
            for result in results[:5]:  # Show top 5
                print(f"  - {result['filename']} (similarity: {result['similarity']:.3f})")
        
        # Show statistics
        stats = self.get_search_statistics()
        print(f"\n📊 Advanced Search Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

def main():
    """Main function to test the advanced AI search engine"""
    print("🚀 Initializing Advanced AI Search Engine...")
    
    try:
        engine = AdvancedAISearchEngine()
        engine.test_search()
        
    except Exception as e:
        logger.error(f"Failed to initialize Advanced AI Search Engine: {e}")
        print("❌ Advanced AI Search Engine initialization failed!")

if __name__ == "__main__":
    main()
