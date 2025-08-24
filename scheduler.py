#!/usr/bin/env python3
"""
Scheduler service for managing continuous transcription and periodic summarization
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

class TranscriptionScheduler:
    """Manages continuous transcription and periodic summarization"""
    
    def __init__(self, 
                 summary_interval_minutes=20,
                 project_dir="/home/abhishek/abhi/logging",
                 language=None,
                 translate_to_english=False):
        self.summary_interval = summary_interval_minutes * 60  # Convert to seconds
        self.project_dir = Path(project_dir)
        self.language = language  # Language code for transcription
        self.translate_to_english = translate_to_english  # Auto-translate to English
        self.is_running = False
        self.transcript_process = None
        self.summary_process = None
        self.summary_thread = None
        
        # Process tracking
        self.processes = {}
        self.pid_file = self.project_dir / "scheduler.pid"
        
        self.setup_logging()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """Setup logging for the scheduler"""
        log_file = self.project_dir / "scheduler.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [SCHEDULER] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def write_pid_file(self):
        """Write process ID to file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"PID {os.getpid()} written to {self.pid_file}")
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {e}")
    
    def remove_pid_file(self):
        """Remove PID file"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("PID file removed")
        except Exception as e:
            self.logger.error(f"Failed to remove PID file: {e}")
    
    def start_transcription(self):
        """Start the continuous transcription process"""
        if self.transcript_process and self.transcript_process.poll() is None:
            self.logger.info("Transcription already running")
            return True
        
        try:
            self.logger.info("Starting continuous transcription...")
            
            # Use Poetry to run the transcription
            cmd = ["poetry", "run", "python", "transcript.py", "--continuous"]
            if self.language:
                cmd.extend(["--language", self.language])
            if self.translate_to_english:
                cmd.append("--translate")
            
            self.transcript_process = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.processes['transcript'] = self.transcript_process.pid
            self.logger.info(f"Transcription started with PID: {self.transcript_process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start transcription: {e}")
            return False
    
    def stop_transcription(self):
        """Stop the transcription process"""
        if not self.transcript_process:
            return True
        
        try:
            self.logger.info("Stopping transcription...")
            
            # Terminate the process group
            os.killpg(os.getpgid(self.transcript_process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                self.transcript_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Transcription didn't stop gracefully, forcing...")
                os.killpg(os.getpgid(self.transcript_process.pid), signal.SIGKILL)
                self.transcript_process.wait()
            
            self.transcript_process = None
            self.processes.pop('transcript', None)
            self.logger.info("Transcription stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop transcription: {e}")
            return False
    
    def run_summary(self):
        """Run summarization (blocking)"""
        try:
            self.logger.info("Running summarization...")
            
            # Get current date for summary generation
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            cmd = [
                "poetry", "run", "python", "generate_summaries_llm.py",
                "--method", "transformers",
                "--date", current_date,
                "--interval", "2"  # Use 2-minute intervals to match daemon frequency
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("✓ Summarization completed successfully")
                # Log some output if available
                if result.stdout.strip():
                    self.logger.info(f"Summary output: {result.stdout[:200]}...")
                return True
            else:
                self.logger.error(f"✗ Summarization failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("✗ Summarization timed out")
            return False
        except Exception as e:
            self.logger.error(f"✗ Error running summarization: {e}")
            return False
    
    def verify_summary_saved(self):
        """Verify that a summary was recently saved to the database"""
        try:
            from db_utils import get_database_manager
            
            db_path = self.project_dir / "transcriptions/transcriptions.db"
            if not db_path.exists():
                self.logger.warning("Database file not found")
                return False
            
            db_manager = get_database_manager(str(db_path))
            count = db_manager.get_recent_summaries_count(minutes_ago=5, method='transformers')
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Error verifying summary save: {e}")
            return False
    
    def summary_scheduler(self):
        """Background thread that runs summarization periodically"""
        self.logger.info(f"Starting summary scheduler (every {self.summary_interval/60:.1f} minutes)")
        
        failed_attempts = 0
        max_failures = 3
        
        while self.is_running:
            try:
                # Wait for interval or until stopped
                for _ in range(self.summary_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
                if self.is_running:
                    success = self.run_summary()
                    
                    if success:
                        failed_attempts = 0
                        if self.verify_summary_saved():
                            self.logger.info("✓ Summary successfully saved to database")
                        else:
                            self.logger.warning("⚠ WARNING: Summary may not have been saved to database properly!")
                    else:
                        failed_attempts += 1
                        self.logger.warning(f"⚠ WARNING: Summary generation failed! (Attempt {failed_attempts}/{max_failures})")
                        
                        if failed_attempts >= max_failures:
                            self.logger.error(f"✗ CRITICAL: Summary generation has failed {max_failures} times in a row!")
                            self.logger.error("✗ Please check Ollama service and database connectivity")
                    
            except Exception as e:
                failed_attempts += 1
                self.logger.error(f"✗ Error in summary scheduler: {e}")
                self.logger.warning(f"⚠ WARNING: Scheduler error! (Attempt {failed_attempts}/{max_failures})")
                time.sleep(60)  # Wait a minute before trying again
    
    def start(self):
        """Start the scheduler service"""
        if self.is_running:
            self.logger.info("Scheduler already running")
            return
        
        self.logger.info("Starting TranscriptionScheduler...")
        self.write_pid_file()
        
        self.is_running = True
        
        # Start continuous transcription
        if not self.start_transcription():
            self.logger.error("Failed to start transcription, exiting")
            self.stop()
            return
        
        # Start summary scheduler thread
        self.summary_thread = threading.Thread(target=self.summary_scheduler, daemon=True)
        self.summary_thread.start()
        
        self.logger.info("Scheduler started successfully")
        
        # Main loop - just keep running and monitor processes
        try:
            while self.is_running:
                # Check if transcription is still running
                if self.transcript_process and self.transcript_process.poll() is not None:
                    self.logger.warning("Transcription process died, restarting...")
                    self.start_transcription()
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the scheduler service"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping TranscriptionScheduler...")
        self.is_running = False
        
        # Stop transcription
        self.stop_transcription()
        
        # Wait for summary thread to finish
        if self.summary_thread and self.summary_thread.is_alive():
            self.logger.info("Waiting for summary thread to finish...")
            self.summary_thread.join(timeout=10)
        
        self.remove_pid_file()
        self.logger.info("Scheduler stopped")
    
    def status(self):
        """Get status of the scheduler and its processes"""
        status_info = {
            'scheduler_running': self.is_running,
            'scheduler_pid': os.getpid() if self.is_running else None,
            'transcription_running': bool(self.transcript_process and self.transcript_process.poll() is None),
            'transcription_pid': self.transcript_process.pid if self.transcript_process else None,
            'summary_interval_minutes': self.summary_interval / 60,
            'processes': self.processes.copy()
        }
        return status_info


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcription Scheduler")
    parser.add_argument('--interval', '-i', type=int, default=20,
                       help='Summary interval in minutes (default: 20)')
    parser.add_argument('--project-dir', '-d', type=str, 
                       default='/home/abhishek/abhi/logging',
                       help='Project directory path')
    parser.add_argument('--language', '-l', type=str, default=None,
                       help='Language: hi (Hindi), en (English), auto/multilingual (auto-detect), or leave empty for multilingual')
    parser.add_argument('--translate', '-t', action='store_true',
                       help='Auto-translate non-English speech to English before saving')
    
    args = parser.parse_args()
    
    scheduler = TranscriptionScheduler(
        summary_interval_minutes=args.interval,
        project_dir=args.project_dir,
        language=args.language,
        translate_to_english=args.translate
    )
    
    try:
        scheduler.start()
    except Exception as e:
        print(f"Failed to start scheduler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
