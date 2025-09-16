#!/usr/bin/env python3
"""
Competition Launcher - Ensures all components are running properly for AI Challenge V
"""

import os
import sys
import time
import json
import subprocess
import threading
import logging
from pathlib import Path
from datetime import datetime
import psutil
import requests
from flask import Flask, jsonify
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('competition_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompetitionLauncher:
    def __init__(self):
        self.processes = {}
        self.health_check_interval = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 10  # seconds
        
        # Component configurations
        self.components = {
            'data_processor': {
                'script': 'data_processor.py',
                'port': None,
                'health_check': self.check_data_processor_health,
                'required': True
            },
            'flask_app': {
                'script': 'app.py',
                'port': 5001,
                'health_check': self.check_flask_health,
                'required': True
            }
        }
        
        # Performance monitoring
        self.performance_stats = {
            'start_time': datetime.now(),
            'total_queries': 0,
            'avg_response_time': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'disk_usage': 0
        }
    
    def check_system_requirements(self):
        """Check if system meets competition requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check available memory (minimum 8GB)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            logger.warning(f"Low memory: {memory_gb:.1f}GB (recommended: 8GB+)")
        
        # Check disk space (minimum 50GB free)
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_gb < 50:
            logger.warning(f"Low disk space: {disk_gb:.1f}GB (recommended: 50GB+)")
        
        # Check required files
        required_files = [
            'app.py',
            'data_processor.py',
            'requirements.txt',
            'templates/home_optimized_v2.html'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Required file missing: {file}")
                return False
        
        logger.info("System requirements check completed")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            # Install from requirements.txt
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True, capture_output=True)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def start_component(self, component_name, config):
        """Start a component process"""
        try:
            logger.info(f"Starting {component_name}...")
            
            # Start the process
            process = subprocess.Popen([
                sys.executable, config['script']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes[component_name] = {
                'process': process,
                'config': config,
                'start_time': datetime.now(),
                'retries': 0
            }
            
            logger.info(f"{component_name} started with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {component_name}: {e}")
            return False
    
    def stop_component(self, component_name):
        """Stop a component process"""
        if component_name in self.processes:
            process_info = self.processes[component_name]
            process = process_info['process']
            
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"{component_name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"{component_name} force killed")
            except Exception as e:
                logger.error(f"Error stopping {component_name}: {e}")
            
            del self.processes[component_name]
    
    def check_data_processor_health(self):
        """Check data processor health"""
        try:
            # Check if database exists and is accessible
            db_path = Path('./cache/processed_files.db')
            if not db_path.exists():
                return False, "Database not found"
            
            # Check database connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM processed_files')
            count = cursor.fetchone()[0]
            conn.close()
            
            # Check if processing is active
            if count == 0:
                return False, "No files processed"
            
            return True, f"Processing {count} files"
            
        except Exception as e:
            return False, f"Database error: {e}"
    
    def check_flask_health(self):
        """Check Flask app health"""
        try:
            response = requests.get('http://localhost:5001/api/status', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('system_ready', False):
                    return True, f"Ready - {data.get('total_images', 0)} images"
                else:
                    return False, "System not ready"
            else:
                return False, f"HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {e}"
    
    def health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                for component_name, process_info in self.processes.items():
                    config = process_info['config']
                    process = process_info['process']
                    
                    # Check if process is still running
                    if process.poll() is not None:
                        logger.warning(f"{component_name} process died")
                        
                        # Restart if retries available
                        if process_info['retries'] < self.max_retries:
                            process_info['retries'] += 1
                            logger.info(f"Restarting {component_name} (attempt {process_info['retries']})")
                            self.stop_component(component_name)
                            time.sleep(self.retry_delay)
                            self.start_component(component_name, config)
                        else:
                            logger.error(f"{component_name} failed after {self.max_retries} retries")
                    
                    # Component-specific health check
                    if config['health_check']:
                        healthy, message = config['health_check']()
                        if not healthy:
                            logger.warning(f"{component_name} health check failed: {message}")
                
                # Update performance stats
                self.update_performance_stats()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def update_performance_stats(self):
        """Update performance statistics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_stats['memory_usage'] = memory.percent
            
            # CPU usage
            cpu = psutil.cpu_percent(interval=1)
            self.performance_stats['cpu_usage'] = cpu
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.performance_stats['disk_usage'] = disk.percent
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def create_competition_config(self):
        """Create competition configuration file"""
        config = {
            'competition_mode': True,
            'max_query_time': 15,  # seconds
            'max_results': 100,
            'cache_timeout': 300,  # 5 minutes
            'batch_size': 128,
            'max_workers': 8,
            'health_check_interval': 30,
            'performance_monitoring': True,
            'auto_recovery': True,
            'log_level': 'INFO'
        }
        
        with open('competition_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Competition configuration created")
    
    def run_competition_tests(self):
        """Run competition readiness tests"""
        logger.info("Running competition tests...")
        
        tests = [
            ("System Requirements", self.check_system_requirements),
            ("Dependencies", self.install_dependencies),
            ("Data Processor", lambda: self.check_data_processor_health()[0]),
            ("Flask App", lambda: self.check_flask_health()[0])
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                logger.info(f"✓ {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results.append((test_name, False))
                logger.error(f"✗ {test_name}: ERROR - {e}")
        
        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        logger.info(f"Competition tests: {passed}/{total} passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! System is ready for competition!")
            return True
        else:
            logger.error("❌ Some tests failed. Please fix issues before competition.")
            return False
    
    def start_all_components(self):
        """Start all required components"""
        logger.info("Starting all components...")
        
        # Start data processor first
        if not self.start_component('data_processor', self.components['data_processor']):
            logger.error("Failed to start data processor")
            return False
        
        # Wait for data processor to initialize
        logger.info("Waiting for data processor to initialize...")
        time.sleep(30)
        
        # Start Flask app
        if not self.start_component('flask_app', self.components['flask_app']):
            logger.error("Failed to start Flask app")
            return False
        
        # Wait for Flask app to start
        logger.info("Waiting for Flask app to start...")
        time.sleep(10)
        
        logger.info("All components started successfully")
        return True
    
    def run(self):
        """Main launcher function"""
        logger.info("🚀 Starting AI Challenge V Competition Launcher")
        
        try:
            # Create competition config
            self.create_competition_config()
            
            # Run tests
            if not self.run_competition_tests():
                logger.error("Competition tests failed. Exiting.")
                return False
            
            # Start components
            if not self.start_all_components():
                logger.error("Failed to start components. Exiting.")
                return False
            
            # Start health check thread
            health_thread = threading.Thread(target=self.health_check_loop, daemon=True)
            health_thread.start()
            
            logger.info("🎯 Competition system is running!")
            logger.info("📊 Monitor performance at: http://localhost:5001")
            logger.info("🔍 Search interface at: http://localhost:5001")
            logger.info("⏹️  Press Ctrl+C to stop")
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
            
        except Exception as e:
            logger.error(f"Launcher error: {e}")
            return False
        
        finally:
            # Cleanup
            for component_name in list(self.processes.keys()):
                self.stop_component(component_name)
            
            logger.info("Competition launcher stopped")
        
        return True

def main():
    """Main function"""
    launcher = CompetitionLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()



