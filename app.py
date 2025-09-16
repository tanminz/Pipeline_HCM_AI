#!/usr/bin/env python3
"""
Advanced HCMC AI Challenge V - Image and Video Retrieval System
Integrated with Advanced AI Search Engine and Multiple AI Models
"""

import os
import json
import logging
import time
from flask import Flask, render_template, request, jsonify, send_file

from PIL import Image, ImageDraw, ImageFont
import io
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
mock_images = []
mock_search_results = []
ai_engine = None
AI_AVAILABLE = True

# Advanced AI models
advanced_ai_engine = None
object_detection_model = None
place_recognition_model = None
ocr_model = None
enhanced_image_analyzer = None

def create_sample_image(image_id, filename):
    """Create a sample image with text"""
    try:
        img = Image.new('RGB', (300, 200), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text = f"ID: {image_id}\n{filename}"
        draw.text((10, 10), text, fill='black', font=font)
        
        return img
    except Exception as e:
        logger.error(f"Error creating sample image: {e}")
        return Image.new('RGB', (300, 200), color='red')

def create_thumbnail(source_path, thumbnail_path, size=(200, 150)):
    """Create thumbnail from source image"""
    try:
        with Image.open(source_path) as img:
            img.thumbnail(size)
            img.save(thumbnail_path, 'JPEG')
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")

def initialize_system():
    """Initialize the system with real competition data and AI engines"""
    global mock_images, mock_search_results, ai_engine, advanced_ai_engine
    
    # Create static directories
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("static/thumbnails", exist_ok=True)
    
    # Load real image metadata from JSON file
    metadata_file = "real_image_metadata.json"
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            mock_images = []
            mock_search_results = []
            
            for image_id_str, data in image_metadata.items():
                image_data = {
                    "id": data['id'],
                    "path": data['web_path'],
                    "filename": image_id_str
                }
                mock_images.append(image_data)
                
                search_data = {
                    "id": data['id'],
                    "path": data['web_path'],
                    "filename": image_id_str,
                    "similarity": 0.95 - (data['id'] * 0.0001),
                    "rank": data['id'] + 1
                }
                mock_search_results.append(search_data)
            
            total_images = len(mock_images)
            logger.info(f"✅ Loaded {total_images} real competition images from metadata file")
            
            # Initialize Advanced AI Search Engine
            if AI_AVAILABLE:
                try:
                    logger.info("🚀 Initializing Advanced AI Search Engine...")
                    from advanced_ai_search import AdvancedAISearchEngine
                    advanced_ai_engine = AdvancedAISearchEngine("fast_image_metadata.json")
                    logger.info("✅ Advanced AI Search Engine initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Advanced AI Search Engine: {e}")
                    advanced_ai_engine = None
                    
                    # Fallback to basic AI engine
                    try:
                        from ai_search_engine import AISearchEngine
                        ai_engine = AISearchEngine("fast_image_metadata.json")
                        logger.info("✅ Basic AI Search Engine initialized as fallback")
                    except Exception as e2:
                        logger.error(f"❌ Failed to initialize Basic AI Search Engine: {e2}")
                        ai_engine = None
            
            # Initialize Enhanced Image Analyzer
            try:
                logger.info("🔄 Initializing Enhanced Image Analyzer...")
                from enhanced_image_analyzer import EnhancedImageAnalyzer
                enhanced_image_analyzer = EnhancedImageAnalyzer()
                logger.info("✅ Enhanced Image Analyzer initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Image Analyzer: {e}")
                enhanced_image_analyzer = None
            
        except Exception as e:
            logger.error(f"Error loading real metadata: {e}")
            total_images = len(mock_images)
            logger.warning("Using fallback mock data")
    else:
        logger.warning("Real metadata file not found, using mock data")
        total_images = len(mock_images)
    
    logger.info(f"🎯 System initialized with {total_images} images")
    # Initialize CSV Export System
    csv_export_system = None
    if enhanced_image_analyzer is not None:
        try:
            csv_export_system = create_csv_export_routes(app, enhanced_image_analyzer)
            logger.info("✅ CSV Export System initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CSV Export System: {e}")
            csv_export_system = None

    return total_images

@app.route('/')
def home():
    """Home page with optimized UI"""
    return render_template('home_optimized_v2.html')

@app.route('/nearby_frames')
def nearby_frames_page():
    """Nearby frames page"""
    return render_template('nearby_frames.html')

@app.route('/api/search')
def api_search():
    """Advanced AI-powered search endpoint with multiple models"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'single')
    k = request.args.get('k', 300, type=int)
    
    if not query:
        return jsonify({'results': [], 'count': 0})
    
    try:
        # Use Enhanced Image Analyzer for better search if available
        if enhanced_image_analyzer is not None and search_type in ['single', 'enhanced']:
            logger.info(f"🔍 Enhanced Analysis Search: '{query}' (type: {search_type}, k: {k})")
            
            # Get image paths from metadata
            image_paths = []
            metadata_file = "fast_image_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    image_metadata = json.load(f)
                
                for image_id_str, data in image_metadata.items():
                    if os.path.exists(data['web_path']):
                        image_paths.append(data['web_path'])
            
            if image_paths:
                # Search using enhanced analysis
                analysis_results = enhanced_image_analyzer.search_by_analysis(query, image_paths, k)
                
                # Format results for API response
                results = []
                for result in analysis_results:
                    image_path = result['image_path']
                    analysis = result['analysis']
                    
                    # Find corresponding metadata
                    for image_id_str, data in image_metadata.items():
                        if data['web_path'] == image_path:
                            results.append({
                                'id': data['id'],
                                'path': data['web_path'],
                                'filename': image_id_str,
                                'similarity': result['score'] / 10.0,  # Normalize score
                                'rank': len(results) + 1,
                                'analysis_summary': analysis.get('text_summary', ''),
                                'detected_objects': analysis.get('objects', {}).get('main_objects', []),
                                'extracted_text': analysis.get('ocr_results', {}).get('combined_text', '')
                            })
                            break
                
                logger.info(f"✅ Enhanced Analysis Search found {len(results)} results")
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'query': query,
                    'strategy': f'enhanced_analysis_{search_type}',
                    'ai_engine': True,
                    'engine_type': 'enhanced_analyzer'
                })
        
        # Use Advanced AI Search Engine if available
        if advanced_ai_engine is not None:
            logger.info(f"🔍 Advanced AI Search: '{query}' (type: {search_type}, k: {k})")
            
            if search_type == 'single':
                results = advanced_ai_engine.search_by_text(query, k)
            elif search_type == 'advanced':
                filters = {}
                if request.args.get('source'):
                    filters['video_folder'] = request.args.get('source')
                if request.args.get('min_similarity'):
                    filters['min_similarity'] = float(request.args.get('min_similarity'))
                results = advanced_ai_engine.advanced_search(query, filters, k)
            else:
                results = advanced_ai_engine.search_by_text(query, k)
            
            logger.info(f"✅ Advanced AI Search found {len(results)} results")
            
            return jsonify({
                'results': results,
                'count': len(results),
                'query': query,
                'strategy': f'advanced_ai_{search_type}',
                'ai_engine': True,
                'engine_type': 'advanced'
            })
        
        # Fallback to basic AI engine
        elif ai_engine is not None:
            logger.info(f"🔍 Basic AI Search: '{query}' (type: {search_type}, k: {k})")
            
            if search_type == 'single':
                results = ai_engine.search_by_text(query, k)
            elif search_type == 'advanced':
                filters = {}
                if request.args.get('source'):
                    filters['source'] = request.args.get('source')
                if request.args.get('min_size'):
                    filters['min_size'] = int(request.args.get('min_size'))
                results = ai_engine.advanced_search(query, filters, k)
            else:
                results = ai_engine.search_by_text(query, k)
            
            logger.info(f"✅ Basic AI Search found {len(results)} results")
            
            return jsonify({
                'results': results,
                'count': len(results),
                'query': query,
                'strategy': f'basic_ai_{search_type}',
                'ai_engine': True,
                'engine_type': 'basic'
            })
        
        # Fallback to search strategies
        else:
            logger.info(f"🔍 Fallback Search: '{query}' (type: {search_type}, k: {k})")
            
            try:
                from search_strategies import SearchStrategies
                strategies = SearchStrategies()
                
                if search_type == 'single':
                    results = strategies.single_search(query, k)
                elif search_type == 'group':
                    results = strategies.group_search(query, k)
                elif search_type == 'local':
                    video_source = request.args.get('video_source', 'L21')
                    results = strategies.local_search(query, video_source, k)
                elif search_type == 'hierarchical':
                    results = strategies.hierarchical_search(query, k1=20, k2=10)
                elif search_type == 'fusion':
                    sub_queries = request.args.getlist('sub_queries[]')
                    weights = request.args.getlist('weights[]')
                    
                    if not sub_queries:
                        sub_queries = query.split()
                        weights = [1.0 / len(sub_queries)] * len(sub_queries)
                    else:
                        weights = [float(w) for w in weights]
                    
                    results = strategies.fusion_search(query, sub_queries, weights, k)
                else:
                    results = strategies.single_search(query, k)
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'query': query,
                    'strategy': search_type,
                    'ai_engine': False,
                    'engine_type': 'strategies'
                })
            
            except Exception as e:
                logger.error(f"Search strategies error: {e}")
                # Final fallback to mock search
                results = []
                query_lower = query.lower()
                
                for i, img in enumerate(mock_images[:k]):
                    if query_lower in img['filename'].lower():
                        results.append({
                            'id': img['id'],
                            'path': img['path'],
                            'filename': img['filename'],
                            'similarity': 0.9 - (i * 0.1),
                            'rank': i + 1
                        })
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'query': query,
                    'strategy': 'mock_fallback',
                    'ai_engine': False,
                    'engine_type': 'mock'
                })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            'results': [],
            'count': 0,
            'error': str(e),
            'query': query
        })

@app.route('/textsearch')
def textsearch():
    """Text search endpoint for web interface"""
    textquery = request.args.get('textquery', '')
    k = request.args.get('k', 300, type=int)  # Default to 300 frames
    
    if not textquery:
        return render_template('home.html', data={'pagefile': [], 'num_page': 0})
    
    try:
        logger.info(f"🔍 Text Search: '{textquery}' (k: {k})")
        
        # Use Advanced AI Search Engine if available
        if advanced_ai_engine is not None:
            results = advanced_ai_engine.search_by_text(textquery, k)
        elif ai_engine is not None:
            results = ai_engine.search_by_text(textquery, k)
        else:
            # Fallback to mock search
            results = []
            query_lower = textquery.lower()
            
            for i, img in enumerate(mock_images[:k]):
                if query_lower in img['filename'].lower():
                    results.append({
                        'id': img['id'],
                        'path': img['path'],
                        'filename': img['filename'],
                        'similarity': 0.9 - (i * 0.1),
                        'rank': i + 1
                    })
        
        # Format results for template
        pagefile = []
        for result in results:
            pagefile.append({
                'id': result['id'],
                'imgpath': result['path'],
                'filename': result['filename'],
                'similarity': result.get('similarity', 0.0),
                'rank': result.get('rank', 0)
            })
        
        # Calculate pagination
        per_page = 300  # Show 300 images per page
        num_page = (len(pagefile) + per_page - 1) // per_page
        
        logger.info(f"✅ Text Search found {len(results)} results, {num_page} pages")
        
        return render_template('home.html', data={
            'pagefile': pagefile,
            'num_page': num_page,
            'query': textquery,
            'total_results': len(results),
            'k': k
        })
        
    except Exception as e:
        logger.error(f"Text search error: {e}")
        return render_template('home.html', data={
            'pagefile': [],
            'num_page': 0,
            'query': textquery,
            'error': str(e)
        })

@app.route('/api/images')
def api_images():
    """Get initial images for display"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 300))
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        images = mock_images[start_idx:end_idx]
        
        return jsonify({
            'images': images,
            'page': page,
            'per_page': per_page,
            'total': len(mock_images)
        })
    except Exception as e:
        logger.error(f"Images API error: {e}")
        return jsonify({
            'images': [],
            'page': 1,
            'per_page': 300,
            'total': 0
        })

@app.route('/api/status')
def api_status():
    """Get system status"""
    try:
        import psutil
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used // (1024 * 1024)
        
        return jsonify({
            'total_images': len(mock_images),
            'memory_usage_mb': memory_usage_mb,
            'ai_engine_available': AI_AVAILABLE,
            'advanced_ai_available': advanced_ai_engine is not None,
            'status': 'ready'
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'total_images': len(mock_images),
            'memory_usage_mb': 0,
            'ai_engine_available': False,
            'advanced_ai_available': False,
            'status': 'error'
        })

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images directly from static/images directory"""
    try:
        # Clean the filename
        filename = filename.replace('\\', '/')  # Handle Windows paths
        
        # Try to find the image in static/images
        image_path = os.path.join("static/images", filename)
        
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        
        # If not found, return 404
        return "Image not found", 404
        
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return "Error serving image", 500

@app.route('/api/thumbnail/<path:filename>')
def api_thumbnail(filename):
    """Serve thumbnail for image by filename"""
    try:
        # Clean the filename
        filename = filename.replace('\\', '/')  # Handle Windows paths
        
        # Try to find the image in static/images
        image_path = os.path.join("static/images", filename)
        
        if os.path.exists(image_path):
            # Return the actual image file
            return send_file(image_path, mimetype='image/jpeg')
        
        # If not found, try to find in our metadata
        for image_data in mock_images:
            if image_data['filename'] == filename:
                metadata_path = image_data['path']
                if os.path.exists(metadata_path):
                    return send_file(metadata_path, mimetype='image/jpeg')
        
        # If still not found, create a placeholder image
        logger.warning(f"Image not found: {filename}, creating placeholder")
        placeholder = create_sample_image(0, filename)
        
        # Convert to bytes
        img_io = io.BytesIO()
        placeholder.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Thumbnail error for {filename}: {e}")
        # Return a simple error image
        error_img = Image.new('RGB', (200, 150), color='red')
        img_io = io.BytesIO()
        error_img.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

@app.route('/api/ai_status')
def api_ai_status():
    """Get AI engine status and statistics"""
    try:
        if advanced_ai_engine is not None:
            stats = advanced_ai_engine.get_search_statistics()
            return jsonify({
                'ai_available': True,
                'status': 'ready',
                'engine_type': 'advanced',
                'statistics': stats
            })
        elif ai_engine is not None:
            stats = ai_engine.get_search_statistics()
            return jsonify({
                'ai_available': True,
                'status': 'ready',
                'engine_type': 'basic',
                'statistics': stats
            })
        else:
            return jsonify({
                'ai_available': False,
                'status': 'not_initialized',
                'engine_type': 'none',
                'error': 'AI engine not available'
            })
    except Exception as e:
        return jsonify({
            'ai_available': False,
            'status': 'error',
            'engine_type': 'error',
            'error': str(e)
        })

@app.route('/api/object_detection/<int:image_id>')
def api_object_detection(image_id):
    """Object detection endpoint using Faster R-CNN"""
    try:
        if image_id < len(mock_images):
            image_data = mock_images[image_id]
            image_path = image_data['path']
            
            if os.path.exists(image_path):
                # TODO: Implement Faster R-CNN object detection
                # For now, return mock object detection results
                objects = [
                    {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
                    {'class': 'car', 'confidence': 0.87, 'bbox': [300, 150, 450, 250]}
                ]
                
                return jsonify({
                    'image_id': image_id,
                    'objects': objects,
                    'model': 'faster_rcnn_mock'
                })
        
        return jsonify({'error': 'Image not found'})
        
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/place_recognition/<int:image_id>')
def api_place_recognition(image_id):
    """Place recognition endpoint using InceptionResNetV2"""
    try:
        if image_id < len(mock_images):
            image_data = mock_images[image_id]
            image_path = image_data['path']
            
            if os.path.exists(image_path):
                # TODO: Implement InceptionResNetV2 place recognition
                # For now, return mock place recognition results
                places = [
                    {'place': 'indoor', 'confidence': 0.92},
                    {'place': 'office', 'confidence': 0.78},
                    {'place': 'building', 'confidence': 0.65}
                ]
                
                return jsonify({
                    'image_id': image_id,
                    'places': places,
                    'model': 'inception_resnet_v2_mock'
                })
        
        return jsonify({'error': 'Image not found'})
        
    except Exception as e:
        logger.error(f"Place recognition error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/ocr/<int:image_id>')
def api_ocr(image_id):
    """OCR endpoint for text extraction"""
    try:
        if image_id < len(mock_images):
            image_data = mock_images[image_id]
            image_path = image_data['path']
            
            if os.path.exists(image_path):
                # Use Enhanced Image Analyzer if available
                if enhanced_image_analyzer is not None:
                    analysis = enhanced_image_analyzer.analyze_image(image_path)
                    if "error" not in analysis:
                        ocr_results = analysis.get("ocr_results", {})
                        return jsonify({
                            'image_id': image_id,
                            'texts': ocr_results.get("text_blocks", []),
                            'combined_text': ocr_results.get("combined_text", ""),
                            'model': 'enhanced_ocr'
                        })
                
                # Fallback to mock OCR results
                text_results = [
                    {'text': 'Sample Text', 'confidence': 0.95, 'bbox': [50, 50, 200, 80]},
                    {'text': 'Another Text', 'confidence': 0.87, 'bbox': [100, 150, 300, 180]}
                ]
                
                return jsonify({
                    'image_id': image_id,
                    'texts': text_results,
                    'model': 'ocr_mock'
                })
        
        return jsonify({'error': 'Image not found'})
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/load_initial_images')
def api_load_initial_images():
    """Load initial images for display"""
    try:
        # Return first 300-500 images for quick display
        initial_count = min(500, len(mock_images))
        initial_images = []
        
        for i in range(initial_count):
            if i < len(mock_images):
                img = mock_images[i]
                initial_images.append({
                    'id': img['id'],
                    'path': img['path'],
                    'filename': img['filename']
                })
        
        return jsonify({
            'images': initial_images,
            'total_count': len(mock_images),
            'loaded_count': len(initial_images)
        })
        
    except Exception as e:
        logger.error(f"Error loading initial images: {e}")
        return jsonify({'images': [], 'error': str(e)})

@app.route('/api/search_nearby_frames')
def api_search_nearby_frames():
    """Search for nearby frames in the same video"""
    frame_id = request.args.get('frame_id', '')
    video_name = request.args.get('video_name', '')
    k = request.args.get('k', 10, type=int)
    
    if not frame_id or not video_name:
        return jsonify({'results': [], 'count': 0, 'error': 'Missing frame_id or video_name'})
    
    try:
        # Extract frame number from frame_id (e.g., "L21_V001/152.jpg" -> 152)
        frame_parts = frame_id.split('/')
        if len(frame_parts) != 2:
            return jsonify({'results': [], 'count': 0, 'error': 'Invalid frame_id format'})
        
        frame_filename = frame_parts[1]
        frame_number_str = frame_filename.replace('.jpg', '')
        
        try:
            frame_number = int(frame_number_str)
        except ValueError:
            return jsonify({'results': [], 'count': 0, 'error': 'Invalid frame number'})
        
        # Find all frames in the same video
        nearby_frames = []
        
        # Load metadata to find frames in the same video
        metadata_file = "fast_image_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            # Find all frames in the same video
            video_frames = []
            for image_id_str, data in image_metadata.items():
                if image_id_str.startswith(video_name + '/'):
                    frame_parts = image_id_str.split('/')
                    if len(frame_parts) == 2:
                        frame_filename = frame_parts[1]
                        frame_num_str = frame_filename.replace('.jpg', '')
                        try:
                            frame_num = int(frame_num_str)
                            video_frames.append({
                                'frame_id': image_id_str,
                                'frame_number': frame_num,
                                'path': data['web_path'],
                                'id': data['id']
                            })
                        except ValueError:
                            continue
            
            # Sort frames by frame number
            video_frames.sort(key=lambda x: x['frame_number'])
            
            # Find the target frame index
            target_index = -1
            for i, frame in enumerate(video_frames):
                if frame['frame_number'] == frame_number:
                    target_index = i
                    break
            
            if target_index != -1:
                # Get nearby frames (k frames before and after)
                start_index = max(0, target_index - k // 2)
                end_index = min(len(video_frames), target_index + k // 2 + 1)
                
                nearby_frames = video_frames[start_index:end_index]
                
                # Format results
                results = []
                for i, frame in enumerate(nearby_frames):
                    results.append({
                        'id': frame['id'],
                        'path': frame['path'],
                        'filename': frame['frame_id'],
                        'frame_number': frame['frame_number'],
                        'is_target': frame['frame_number'] == frame_number,
                        'distance': abs(frame['frame_number'] - frame_number),
                        'rank': i + 1
                    })
                
                logger.info(f"Found {len(results)} nearby frames for {frame_id}")
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'target_frame': frame_number,
                    'video_name': video_name,
                    'nearby_range': f"{start_index}-{end_index-1}"
                })
        
        return jsonify({'results': [], 'count': 0, 'error': 'Video not found'})
        
    except Exception as e:
        logger.error(f"Error searching nearby frames: {e}")
        return jsonify({'results': [], 'count': 0, 'error': str(e)})

@app.route('/api/frame_info')
def api_frame_info():
    """Get frame information for UI display"""
    frame_id = request.args.get('frame_id', '')
    
    if not frame_id:
        return jsonify({'error': 'Missing frame_id'})
    
    try:
        # Parse frame_id to get video name and frame number
        frame_parts = frame_id.split('/')
        if len(frame_parts) != 2:
            return jsonify({'error': 'Invalid frame_id format'})
        
        video_name = frame_parts[0]
        frame_filename = frame_parts[1]
        frame_number_str = frame_filename.replace('.jpg', '')
        
        try:
            frame_number = int(frame_number_str)
        except ValueError:
            return jsonify({'error': 'Invalid frame number'})
        
        # Load metadata to get frame info
        metadata_file = "fast_image_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            if frame_id in image_metadata:
                frame_data = image_metadata[frame_id]
                
                # Count total frames in this video
                video_frames = []
                for img_id, data in image_metadata.items():
                    if img_id.startswith(video_name + '/'):
                        video_frames.append(img_id)
                
                video_frames.sort()
                total_frames = len(video_frames)
                
                return jsonify({
                    'frame_id': frame_id,
                    'video_name': video_name,
                    'frame_number': frame_number,
                    'total_frames': total_frames,
                    'frame_position': video_frames.index(frame_id) + 1 if frame_id in video_frames else 0,
                    'path': frame_data['web_path'],
                    'id': frame_data['id']
                })
        
        return jsonify({'error': 'Frame not found'})
        
    except Exception as e:
        logger.error(f"Error getting frame info: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/search_nearby_frames_ui')
def api_search_nearby_frames_ui():
    """Search nearby frames with UI-friendly response and pagination"""
    frame_id = request.args.get('frame_id', '')
    k = request.args.get('k', 300, type=int)  # Default to 300 frames
    page = request.args.get('page', 1, type=int)  # Pagination
    per_page = request.args.get('per_page', 300, type=int)  # Frames per page - default 300
    
    # If per_page is very large, show all frames
    show_all_frames = per_page >= 10000
    
    if not frame_id:
        return jsonify({'results': [], 'count': 0, 'error': 'Missing frame_id'})
    
    try:
        # Parse frame_id
        frame_parts = frame_id.split('/')
        if len(frame_parts) != 2:
            return jsonify({'results': [], 'count': 0, 'error': 'Invalid frame_id format'})
        
        video_name = frame_parts[0]
        frame_filename = frame_parts[1]
        frame_number_str = frame_filename.replace('.jpg', '')
        
        try:
            frame_number = int(frame_number_str)
        except ValueError:
            return jsonify({'results': [], 'count': 0, 'error': 'Invalid frame number'})
        
        # Load metadata
        metadata_file = "fast_image_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            # Find all frames in the same video
            video_frames = []
            for image_id_str, data in image_metadata.items():
                if image_id_str.startswith(video_name + '/'):
                    frame_parts = image_id_str.split('/')
                    if len(frame_parts) == 2:
                        frame_filename = frame_parts[1]
                        frame_num_str = frame_filename.replace('.jpg', '')
                        try:
                            frame_num = int(frame_num_str)
                            video_frames.append({
                                'frame_id': image_id_str,
                                'frame_number': frame_num,
                                'path': data['web_path'],
                                'id': data['id']
                            })
                        except ValueError:
                            continue
            
            # Sort frames by frame number
            video_frames.sort(key=lambda x: x['frame_number'])
            
            # Find the target frame index
            target_index = -1
            for i, frame in enumerate(video_frames):
                if frame['frame_number'] == frame_number:
                    target_index = i
                    break
            
            if target_index != -1:
                total_frames = len(video_frames)
                
                if show_all_frames:
                    # Show all frames without pagination
                    page_frames = video_frames
                    start_index = 0
                    end_index = total_frames
                    page = 1
                    total_pages = 1
                else:
                    # If page is 1, automatically go to the page containing target frame
                    if page == 1:
                        target_position = target_index + 1  # Convert to 1-based index
                        page = (target_position + per_page - 1) // per_page
                    
                    # Calculate pagination
                    total_pages = (total_frames + per_page - 1) // per_page
                    start_index = (page - 1) * per_page
                    end_index = min(start_index + per_page, total_frames)
                    
                    # Get frames for current page
                    page_frames = video_frames[start_index:end_index]
                
                # Format results for UI
                results = []
                for i, frame in enumerate(page_frames):
                    results.append({
                        'id': frame['id'],
                        'path': frame['path'],
                        'filename': frame['frame_id'],
                        'frame_number': frame['frame_number'],
                        'is_target': frame['frame_number'] == frame_number,
                        'distance': abs(frame['frame_number'] - frame_number),
                        'rank': start_index + i + 1,
                        'thumbnail_url': f"/api/thumbnail/{frame['frame_id']}"
                    })
                
                if show_all_frames:
                    logger.info(f"Found {len(results)} frames (ALL FRAMES MODE - total: {total_frames})")
                    
                    return jsonify({
                        'results': results,
                        'count': len(results),
                        'total_frames': total_frames,
                        'target_frame': frame_number,
                        'video_name': video_name,
                        'target_position': target_index + 1,
                        'target_page': 1,
                        'pagination': {
                            'current_page': 1,
                            'total_pages': 1,
                            'per_page': total_frames,
                            'start_index': 1,
                            'end_index': total_frames,
                            'has_prev': False,
                            'has_next': False
                        }
                    })
                else:
                    logger.info(f"Found {len(results)} frames for page {page} of {total_pages} (total: {total_frames}, target on page: {(target_index + 1 + per_page - 1) // per_page})")
                    
                    return jsonify({
                        'results': results,
                        'count': len(results),
                        'total_frames': total_frames,
                        'target_frame': frame_number,
                        'video_name': video_name,
                        'target_position': target_index + 1,
                        'target_page': (target_index + 1 + per_page - 1) // per_page,
                        'pagination': {
                            'current_page': page,
                            'total_pages': total_pages,
                            'per_page': per_page,
                            'start_index': start_index + 1,
                            'end_index': end_index,
                            'has_prev': page > 1,
                            'has_next': page < total_pages
                        }
                    })
        
        return jsonify({'results': [], 'count': 0, 'error': 'Video not found'})
        
    except Exception as e:
        logger.error(f"Error searching nearby frames: {e}")
        return jsonify({'results': [], 'count': 0, 'error': str(e)})

@app.route('/api/translate_test')
def api_translate_test():
    """Test Vietnamese translation endpoint"""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'Missing query parameter'})
    
    try:
        # Test enhanced translator
        try:
            from enhanced_vietnamese_translator import translate_vietnamese_query, get_query_suggestions, analyze_vietnamese_query
            
            translated = translate_vietnamese_query(query)
            suggestions = get_query_suggestions(query)
            analysis = analyze_vietnamese_query(query)
            
            return jsonify({
                'original_query': query,
                'translated_query': translated,
                'suggestions': suggestions,
                'analysis': analysis,
                'translator': 'enhanced'
            })
        except ImportError:
            # Fallback to basic translation
            if advanced_ai_engine is not None:
                translated = advanced_ai_engine.translate_vietnamese_to_english(query)
                return jsonify({
                    'original_query': query,
                    'translated_query': translated,
                    'translator': 'basic'
                })
            else:
                return jsonify({
                    'original_query': query,
                    'translated_query': query,
                    'translator': 'none'
                })
                
    except Exception as e:
        logger.error(f"Translation test error: {e}")
        return jsonify({
            'original_query': query,
            'error': str(e),
            'translator': 'error'
        })

@app.route('/api/search_with_translation')
def api_search_with_translation():
    """Search with detailed translation information"""
    query = request.args.get('q', '')
    k = request.args.get('k', 20, type=int)
    
    if not query:
        return jsonify({'results': [], 'count': 0, 'error': 'Missing query parameter'})
    
    try:
        # Get translation analysis
        translation_info = {}
        try:
            from enhanced_vietnamese_translator import translate_vietnamese_query, analyze_vietnamese_query
            translated_query = translate_vietnamese_query(query)
            analysis = analyze_vietnamese_query(query)
            translation_info = {
                'original': query,
                'translated': translated_query,
                'analysis': analysis
            }
        except ImportError:
            if advanced_ai_engine is not None:
                translated_query = advanced_ai_engine.translate_vietnamese_to_english(query)
                translation_info = {
                    'original': query,
                    'translated': translated_query
                }
            else:
                translated_query = query
                translation_info = {
                    'original': query,
                    'translated': query
                }
        
        # Perform search
        if advanced_ai_engine is not None:
            results = advanced_ai_engine.search_by_text(query, k)
        elif ai_engine is not None:
            results = ai_engine.search_by_text(query, k)
        else:
            # Fallback search
            results = []
            query_lower = query.lower()
            
            for i, img in enumerate(mock_images[:k]):
                if query_lower in img['filename'].lower():
                    results.append({
                        'id': img['id'],
                        'path': img['path'],
                        'filename': img['filename'],
                        'similarity': 0.9 - (i * 0.1),
                        'rank': i + 1
                    }) 
        
        return jsonify({
            'results': results,
            'count': len(results),
            'translation_info': translation_info,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Search with translation error: {e}")
        return jsonify({
            'results': [],
            'count': 0,
            'error': str(e),
            'query': query
        })

@app.route('/api/enhanced_analysis/<int:image_id>')
def api_enhanced_analysis(image_id):
    """Enhanced image analysis endpoint with OCR, objects, and metadata"""
    try:
        if image_id < len(mock_images):
            image_data = mock_images[image_id]
            image_path = image_data['path']
            
            if os.path.exists(image_path):
                # Use Enhanced Image Analyzer if available
                if enhanced_image_analyzer is not None:
                    analysis = enhanced_image_analyzer.analyze_image(image_path)
                    if "error" not in analysis:
                        return jsonify({
                            'image_id': image_id,
                            'analysis': analysis,
                            'model': 'enhanced_analyzer'
                        })
                
                # Fallback to basic analysis
                return jsonify({
                    'image_id': image_id,
                    'analysis': {
                        'image_info': {'width': 640, 'height': 480, 'format': 'JPEG'},
                        'ocr_results': {'combined_text': 'Sample text'},
                        'objects': {'main_objects': ['person', 'car']},
                        'scene_classification': {'scene_type': 'outdoor'},
                        'text_summary': 'Sample analysis',
                        'search_keywords': ['sample', 'text']
                    },
                    'model': 'basic_analyzer'
                })
        
        return jsonify({'error': 'Image not found'})
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/image_info/<int:image_id>')
def api_image_info(image_id):
    """Get basic image information"""
    try:
        if image_id < len(mock_images):
            image_data = mock_images[image_id]
            image_path = image_data['path']
            
            if os.path.exists(image_path):
                # Get image info using PIL
                from PIL import Image
                with Image.open(image_path) as img:
                    image_info = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'filename': os.path.basename(image_path),
                        'size_bytes': os.path.getsize(image_path),
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode,
                        'aspect_ratio': img.width / img.height if img.height > 0 else 0
                    }
                
                return jsonify(image_info)
        
        return jsonify({'error': 'Image not found'})
        
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/search_with_analysis')
def api_search_with_analysis():
    """Search with enhanced image analysis"""
    query = request.args.get('q', '')
    k = request.args.get('k', 20, type=int)
    
    if not query:
        return jsonify({'results': [], 'count': 0, 'error': 'Missing query parameter'})
    
    try:
        if enhanced_image_analyzer is not None:
            # Get image paths from metadata
            image_paths = []
            metadata_file = "fast_image_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    image_metadata = json.load(f)
                
                for image_id_str, data in image_metadata.items():
                    if os.path.exists(data['web_path']):
                        image_paths.append(data['web_path'])
            
            if image_paths:
                # Search using enhanced analysis
                analysis_results = enhanced_image_analyzer.search_by_analysis(query, image_paths, k)
                
                # Format results for API response
                results = []
                for result in analysis_results:
                    image_path = result['image_path']
                    analysis = result['analysis']
                    
                    # Find corresponding metadata
                    for image_id_str, data in image_metadata.items():
                        if data['web_path'] == image_path:
                            results.append({
                                'id': data['id'],
                                'path': data['web_path'],
                                'filename': image_id_str,
                                'similarity': result['score'] / 10.0,  # Normalize score
                                'rank': len(results) + 1,
                                'analysis_summary': analysis.get('text_summary', ''),
                                'detected_objects': analysis.get('objects', {}).get('main_objects', []),
                                'extracted_text': analysis.get('ocr_results', {}).get('combined_text', '')
                            })
                            break
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'query': query,
                    'strategy': 'enhanced_analysis',
                    'ai_engine': True,
                    'engine_type': 'enhanced_analyzer'
                })
        
        # Fallback to regular search
        if advanced_ai_engine is not None:
            results = advanced_ai_engine.search_by_text(query, k)
        elif ai_engine is not None:
            results = ai_engine.search_by_text(query, k)
        else:
            results = []
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query,
            'strategy': 'fallback_search',
            'ai_engine': False,
            'engine_type': 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Search with analysis error: {e}")
        return jsonify({
            'results': [],
            'count': 0,
            'error': str(e),
            'query': query
        })

@app.route('/api/analyzer_status')
def api_analyzer_status():
    """Get enhanced analyzer status and statistics"""
    try:
        if enhanced_image_analyzer is not None:
            stats = enhanced_image_analyzer.get_analysis_statistics()
            return jsonify({
                'analyzer_available': True,
                'status': 'ready',
                'statistics': stats
            })
        else:
            return jsonify({
                'analyzer_available': False,
                'status': 'not_initialized',
                'error': 'Enhanced analyzer not available'
            })
    except Exception as e:
        return jsonify({
            'analyzer_available': False,
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    # Initialize system
    total_images = initialize_system()
    
    # Start Flask app
    logger.info(f"🚀 Starting Flask app with {total_images} images...")
    app.run(debug=True, host='0.0.0.0', port=5000)
