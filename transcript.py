import whisper
import sounddevice as sd

import threading
import queue
import numpy as np
from datetime import datetime, timedelta

import os
import time
from collections import deque
import logging
import signal
import sys
from db_utils import get_database_manager

# Translation imports
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False

class RealTimeWhisperTranscriber:
    def __init__(self, 
                 model_size="medium",  # base, small, medium, large
                 chunk_duration=5,   # seconds of audio per transcription
                 sample_rate=16000,
                 channels=1,
                 output_dir="transcriptions",
                 language=None,      # Language code ('hi', 'en', 'auto' for auto-detection, or None for multilingual)
                 translate_to_english=False):  # Auto-translate non-English to English
        
        self.model_size = model_size
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_dir = output_dir
        self.language = language  # Store language for transcription
        self.translate_to_english = translate_to_english
        
        # Audio settings
        self.chunk_size = int(sample_rate * chunk_duration)
        self.dtype = np.float32
        
        # Queues for audio processing
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Threading control
        self.is_recording = False
        self.is_processing = False
        
        # Audio buffer for continuous recording
        self.audio_buffer = deque(maxlen=self.chunk_size * 2)  # 2x buffer for overlap
        
        # Initialize components
        self.setup_logging()
        self.setup_directories()
        self.setup_database()
        self.load_whisper_model()
        self.setup_audio()
        self.setup_translator()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transcription.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_database(self):
        """Setup database using the improved database manager"""
        self.db_path = os.path.join(self.output_dir, "transcriptions.db")
        self.db_manager = get_database_manager(self.db_path)
        
    def load_whisper_model(self):
        """Load Whisper model"""
        self.logger.info(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)
        self.logger.info("Whisper model loaded successfully")
        
    def setup_audio(self):
        """Initialize sounddevice"""
        # Get default input device info
        try:
            default_device = sd.default.device[0]  # input device
            device_info = sd.query_devices(default_device)
            self.logger.info(f"Using audio device: {device_info['name']}")
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
            self.logger.info("Using default audio device")
        
        # Set sounddevice defaults
        sd.default.samplerate = self.sample_rate
        sd.default.channels = self.channels
        sd.default.dtype = self.dtype
        
    def setup_translator(self):
        """Setup translation service if enabled"""
        self.translator = None
        
        if not self.translate_to_english:
            self.logger.info("Translation disabled")
            return
        
        # Try to initialize translator
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translator = Translator()
                self.translation_method = "googletrans"
                self.logger.info("✓ Google Translate (googletrans) initialized for auto-translation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize googletrans: {e}")
        
        if self.translator is None and DEEP_TRANSLATOR_AVAILABLE:
            try:
                self.translator = GoogleTranslator(source='auto', target='en')
                self.translation_method = "deep-translator"
                self.logger.info("✓ Deep Translator initialized for auto-translation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize deep-translator: {e}")
        
        if self.translator is None:
            self.logger.warning("⚠ No translation service available. Install googletrans or deep-translator")
            self.logger.warning("  pip install googletrans==4.0.0rc1 deep-translator")
            self.translate_to_english = False  # Disable translation
    
    def translate_text_to_english(self, text, detected_language):
        """Translate text to English if it's not already English"""
        if not self.translate_to_english or not self.translator:
            return text, None
        
        # Skip translation if already English
        if detected_language.lower() in ['en', 'english']:
            return text, None
        
        try:
            if self.translation_method == "googletrans":
                # Using googletrans
                result = self.translator.translate(text, dest='en')
                translated_text = result.text
                confidence = getattr(result, 'confidence', None)
                
                self.logger.info(f"🔤 Translated {detected_language.upper()}→EN: \"{text[:30]}...\" → \"{translated_text[:30]}...\"")
                return translated_text, confidence
                
            elif self.translation_method == "deep-translator":
                # Using deep-translator
                translated_text = self.translator.translate(text)
                
                self.logger.info(f"🔤 Translated {detected_language.upper()}→EN: \"{text[:30]}...\" → \"{translated_text[:30]}...\"")
                return translated_text, None
                
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return text, None  # Return original text on failure
            
        return text, None
        
    def audio_recording_thread(self):
        """Thread function for recording audio using InputStream"""
        self.logger.info("Audio recording thread started")
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio input stream"""
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            
            if self.is_recording and indata is not None:
                # Flatten audio data if needed
                audio_data = indata.flatten() if len(indata.shape) > 1 else indata.copy()
                self.audio_buffer.extend(audio_data)
                
                # Check if we have enough data for transcription
                if len(self.audio_buffer) >= self.chunk_size:
                    chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                    self.audio_queue.put((chunk, datetime.now()))
                    
                    # Remove processed data (with overlap)
                    overlap_size = self.chunk_size // 4  # 25% overlap
                    for _ in range(self.chunk_size - overlap_size):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
        
        try:
            # Use InputStream for non-blocking audio capture
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                # Keep the stream alive while recording
                while self.is_recording:
                    time.sleep(0.1)  # Check every 100ms
                    
        except Exception as e:
            if self.is_recording:  # Only log if we're still supposed to be recording
                self.logger.error(f"Error in audio recording: {e}")
        
        self.logger.info("Audio recording thread stopped")
    
    def audio_processor_thread(self):
        """Thread function for processing audio chunks"""
        self.logger.info("Audio processor thread started")
        
        while self.is_processing:
            try:
                # Get audio chunk from queue (with timeout)
                audio_chunk, timestamp = self.audio_queue.get(timeout=1.0)
                
                # Audio chunk processed (no saving needed - using database only)
                
                # Transcribe audio
                start_time = time.time()
                # Configure transcription options for multilingual support
                transcribe_options = {"fp16": False}
                
                # Handle language settings
                if self.language and self.language.lower() not in ['auto', 'multilingual']:
                    # Use specific language
                    transcribe_options["language"] = self.language
                    self.logger.debug(f"Using fixed language: {self.language}")
                else:
                    # Use auto-detection (Whisper's default behavior)
                    self.logger.debug("Using automatic language detection")
                
                result = self.model.transcribe(audio_chunk, **transcribe_options)
                transcription_time = time.time() - start_time
                
                # Extract transcription, confidence, and detected language
                text = result["text"].strip()
                detected_language = result.get("language", "unknown")
                
                # Log detected language for multilingual mode
                if not self.language or self.language.lower() in ['auto', 'multilingual']:
                    if text:  # Only log if there's actual text
                        self.logger.info(f"🌐 Detected language: {detected_language.upper()} - \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                
                # Translate to English if enabled and text is not already English
                original_text = text
                translation_confidence = None
                if text:  # Only translate non-empty text
                    text, translation_confidence = self.translate_text_to_english(text, detected_language)
                
                # Whisper doesn't provide confidence scores directly, estimate from segments
                segments = result.get("segments", [])
                if segments:
                    avg_confidence = np.mean([segment.get("avg_logprob", -1.0) for segment in segments])
                else:
                    avg_confidence = -1.0  # Default confidence for empty segments
                
                if text:  # Only process non-empty transcriptions
                    self.transcription_queue.put({
                        "timestamp": timestamp,
                        "text": text,  # This will be English (translated if necessary)
                        "original_text": original_text,  # Original text in detected language
                        "confidence": avg_confidence,
                        "duration": transcription_time,
                        "audio_file": audio_filename,
                        "detected_language": detected_language,
                        "translation_confidence": translation_confidence,
                        "was_translated": text != original_text
                    })
                    
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing: {e}")
                
    def transcription_handler_thread(self):
        """Thread function for handling transcriptions"""
        self.logger.info("Transcription handler thread started")
        
        while self.is_processing:
            try:
                # Get transcription from queue
                transcription = self.transcription_queue.get(timeout=1.0)
                
                # Save to database
                self.save_transcription(transcription)
                

                
                # Print to console (optional)
                self.print_transcription(transcription)
                
                self.transcription_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in transcription handling: {e}")
                
    def save_transcription(self, transcription):
        """Save transcription to database using the database manager"""
        try:
            self.db_manager.save_transcription(transcription)
        except Exception as e:
            self.logger.error(f"Failed to save transcription to database: {e}")
        

            
    def print_transcription(self, transcription):
        """Print transcription to console"""
        timestamp_str = transcription["timestamp"].strftime("%H:%M:%S")
        confidence_str = f"{transcription['confidence']:.2f}"
        print(f"[{timestamp_str}] ({confidence_str}): {transcription['text']}")
        
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.info("Interrupt signal received, stopping transcription...")
        self.stop_transcription()
        sys.exit(0)
        
    def start_transcription(self):
        """Start the real-time transcription"""
        self.logger.info("Starting real-time transcription...")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.is_recording = True
        self.is_processing = True
        
        # Start processing threads (not daemon so we can properly join them)
        self.recording_thread = threading.Thread(target=self.audio_recording_thread, daemon=False)
        self.audio_thread = threading.Thread(target=self.audio_processor_thread, daemon=False)
        self.transcription_thread = threading.Thread(target=self.transcription_handler_thread, daemon=False)
        
        self.recording_thread.start()
        self.audio_thread.start()
        self.transcription_thread.start()
        
        self.logger.info("Transcription started. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive and responsive
            while self.is_recording and self.is_processing:
                time.sleep(0.1)  # More responsive checking
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, stopping transcription...")
            self.stop_transcription()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stop_transcription()
            
    def stop_transcription(self):
        """Stop the real-time transcription"""
        self.logger.info("Stopping transcription...")
        
        # Set flags to stop threads
        self.is_recording = False
        self.is_processing = False
        
        # Stop sounddevice streams
        try:
            sd.stop()
        except:
            pass  # Ignore errors if nothing is playing/recording
        
        # Wait for threads to finish with a reasonable timeout
        threads_to_join = [
            ('recording_thread', 'recording'),
            ('audio_thread', 'audio processing'), 
            ('transcription_thread', 'transcription handling')
        ]
        
        for thread_attr, thread_name in threads_to_join:
            if hasattr(self, thread_attr):
                thread = getattr(self, thread_attr)
                if thread.is_alive():
                    self.logger.info(f"Waiting for {thread_name} thread to stop...")
                    thread.join(timeout=3.0)  # 3 second timeout
                    
                    if thread.is_alive():
                        self.logger.warning(f"{thread_name.title()} thread did not stop gracefully")
                    else:
                        self.logger.info(f"{thread_name.title()} thread stopped")
        
        # Clear any remaining items in queues
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.transcription_queue.empty():
                self.transcription_queue.get_nowait()
        except:
            pass
            
        self.logger.info("Transcription stopped.")
        
    def get_daily_summary(self, date=None):
        """Get summary of transcriptions for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            return self.db_manager.get_transcriptions_for_date(date)
        except Exception as e:
            self.logger.error(f"Failed to get daily summary: {e}")
            return []
    
    def group_transcriptions_by_interval(self, interval_minutes=10, date=None):
        """Group transcriptions by time intervals (default: 10 minutes)"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            results = self.db_manager.get_transcriptions_for_date(date)
        except Exception as e:
            self.logger.error(f"Failed to get transcriptions for grouping: {e}")
            return []
        
        if not results:
            return []
        
        grouped_data = []
        current_group = []
        current_start_time = None
        
        for row in results:
            timestamp_str, text, confidence = row
            timestamp = datetime.fromisoformat(timestamp_str)
            
            if current_start_time is None:
                current_start_time = timestamp.replace(second=0, microsecond=0)
                # Round down to nearest 10-minute interval
                current_start_time = current_start_time.replace(minute=(current_start_time.minute // interval_minutes) * interval_minutes)
            
            current_end_time = current_start_time + timedelta(minutes=interval_minutes)
            
            if timestamp < current_end_time:
                current_group.append({
                    'timestamp': timestamp,
                    'text': text,
                    'confidence': confidence
                })
            else:
                # Save current group and start new one
                if current_group:
                    grouped_data.append({
                        'start_time': current_start_time,
                        'end_time': current_end_time,
                        'transcriptions': current_group
                    })
                
                # Start new group
                current_start_time = timestamp.replace(second=0, microsecond=0)
                current_start_time = current_start_time.replace(minute=(current_start_time.minute // interval_minutes) * interval_minutes)
                current_end_time = current_start_time + timedelta(minutes=interval_minutes)
                current_group = [{
                    'timestamp': timestamp,
                    'text': text,
                    'confidence': confidence
                }]
        
        # Add the last group
        if current_group:
            grouped_data.append({
                'start_time': current_start_time,
                'end_time': current_end_time,
                'transcriptions': current_group
            })
        
        return grouped_data
    
    def generate_summary_text(self, transcriptions):
        """Generate a summary from a list of transcriptions"""
        if not transcriptions:
            return ""
        
        # Combine all text
        combined_text = " ".join([t['text'] for t in transcriptions])
        
        # Simple extractive summarization - could be enhanced with more sophisticated NLP
        sentences = combined_text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If few sentences, return all
        if len(sentences) <= 3:
            return combined_text.strip()
        
        # Simple approach: take first, middle, and key sentences
        # In a production system, you might use proper summarization models
        word_count = len(combined_text.split())
        
        if word_count < 50:
            return combined_text.strip()
        elif word_count < 200:
            # Take first 2 and last sentence
            summary_sentences = sentences[:2]
            if len(sentences) > 2:
                summary_sentences.append(sentences[-1])
            return ". ".join(summary_sentences) + "."
        else:
            # Take first, middle, and last sentences for longer content
            mid_idx = len(sentences) // 2
            summary_sentences = [sentences[0], sentences[mid_idx], sentences[-1]]
            return ". ".join(summary_sentences) + "."
    
    def save_summary(self, start_time, end_time, transcriptions):
        """Save a summary to the summaries table"""
        if not transcriptions:
            return
        
        # Generate summary
        summary_text = self.generate_summary_text(transcriptions)
        
        # Calculate statistics
        word_count = len(summary_text.split())
        transcription_count = len(transcriptions)
        avg_confidence = sum(t['confidence'] for t in transcriptions) / transcription_count
        created_at = datetime.now()
        
        # Save to database using database manager
        try:
            self.db_manager.save_summary(
                start_time, end_time, summary_text, word_count,
                transcription_count, avg_confidence, created_at,
                method="simple", model_name=None
            )
            self.logger.info(f"Saved summary for {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
    
    def generate_daily_summaries(self, date=None, interval_minutes=10):
        """Generate summaries for a specific date grouped by intervals"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Generating {interval_minutes}-minute summaries for {date}")
        
        # Get grouped transcriptions
        grouped_data = self.group_transcriptions_by_interval(interval_minutes, date)
        
        if not grouped_data:
            self.logger.info(f"No transcriptions found for {date}")
            return
        
        summaries_created = 0
        
        for group in grouped_data:
            start_time = group['start_time']
            end_time = group['end_time']
            transcriptions = group['transcriptions']
            
            # Check if summary already exists for this interval
            try:
                if not self.db_manager.check_summary_exists(start_time, end_time, "simple"):
                    # Save summary
                    self.save_summary(start_time, end_time, transcriptions)
                    summaries_created += 1
                else:
                    self.logger.debug(f"Summary already exists for {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
            except Exception as e:
                self.logger.error(f"Failed to process summary for {start_time.strftime('%H:%M')}: {e}")
        
        self.logger.info(f"Created {summaries_created} new summaries for {date}")
        return summaries_created


def main():
    """Main function to run the transcriber"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time transcription with Whisper")
    parser.add_argument('--model', '-m', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--chunk-duration', '-c', type=int, default=60,
                       help='Seconds per transcription chunk')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously (daemon mode)')
    parser.add_argument('--output-dir', '-o', type=str, default='transcriptions',
                       help='Output directory for transcriptions')
    parser.add_argument('--language', '-l', type=str, default=None,
                       help='Language code: hi (Hindi), en (English), auto/multilingual (auto-detect), or leave empty for multilingual support')
    parser.add_argument('--translate', '-t', action='store_true',
                       help='Auto-translate non-English speech to English before saving')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "model_size": args.model,
        "chunk_duration": args.chunk_duration,
        "sample_rate": 16000,  # Audio sample rate
        "channels": 1,         # Mono audio
        "output_dir": args.output_dir,
        "language": args.language,
        "translate_to_english": args.translate
    }
    
    # Create and start transcriber
    transcriber = RealTimeWhisperTranscriber(**config)
    
    if args.continuous:
        transcriber.logger.info("Starting in continuous mode...")
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            transcriber.logger.info(f"Received signal {signum}, shutting down...")
            transcriber.stop_transcription()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                try:
                    transcriber.start_transcription()
                except Exception as e:
                    transcriber.logger.error(f"Transcription error: {e}")
                    transcriber.logger.info("Restarting in 10 seconds...")
                    time.sleep(10)
                    # Restart the transcriber
                    transcriber = RealTimeWhisperTranscriber(**config)
        except KeyboardInterrupt:
            transcriber.logger.info("Received keyboard interrupt")
        finally:
            transcriber.stop_transcription()
    else:
        # Normal single-run mode
        try:
            transcriber.start_transcription()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            transcriber.stop_transcription()


if __name__ == "__main__":
    main()