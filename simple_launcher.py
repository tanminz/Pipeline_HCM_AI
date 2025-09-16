#!/usr/bin/env python3
"""
Simple Launcher - Basic system startup without complex features
"""

import os
import sys
import time
import subprocess
import threading
import logging
from pathlib import Path

# Configure logging without special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_launcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleLauncher:
    def __init__(self):
        self.processes = {}
        self.stop_launcher = False
        
    def check_files(self):
        """Check if required files exist"""
        required_files = [
            'app.py',
            'templates/home_optimized_v2.html'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Required file missing: {file}")
                return False
        
        logger.info("All required files found")
        return True
    
    def start_flask_app(self):
        """Start Flask application"""
        try:
            logger.info("Starting Flask app...")
            
            # Start Flask app
            process = subprocess.Popen([
                sys.executable, 'app.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['flask_app'] = {
                'process': process,
                'start_time': time.time()
            }
            
            logger.info(f"Flask app started with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Flask app: {e}")
            return False
    
    def check_flask_health(self):
        """Check if Flask app is running"""
        if 'flask_app' not in self.processes:
            return False
        
        process = self.processes['flask_app']['process']
        return process.poll() is None
    
    def stop_all(self):
        """Stop all processes"""
        for name, info in self.processes.items():
            process = info['process']
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"{name} stopped")
            except:
                process.kill()
                logger.warning(f"{name} force killed")
        
        self.processes.clear()
    
    def run(self):
        """Main launcher function"""
        logger.info("Starting Simple Launcher")
        
        try:
            # Check files
            if not self.check_files():
                logger.error("File check failed. Exiting.")
                return False
            
            # Start Flask app
            if not self.start_flask_app():
                logger.error("Failed to start Flask app. Exiting.")
                return False
            
            # Wait for Flask to start
            logger.info("Waiting for Flask app to start...")
            time.sleep(10)
            
            logger.info("System is running!")
            logger.info("Search interface at: http://localhost:5001")
            logger.info("Press Ctrl+C to stop")
            
            # Keep main thread alive
            try:
                while not self.stop_launcher:
                    # Check if Flask is still running
                    if not self.check_flask_health():
                        logger.error("Flask app died. Restarting...")
                        self.stop_all()
                        if not self.start_flask_app():
                            logger.error("Failed to restart Flask app")
                            break
                        time.sleep(10)
                    
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
            
        except Exception as e:
            logger.error(f"Launcher error: {e}")
            return False
        
        finally:
            self.stop_all()
            logger.info("Simple launcher stopped")
        
        return True

def main():
    """Main function"""
    launcher = SimpleLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()




