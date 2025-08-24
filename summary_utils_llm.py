#!/usr/bin/env python3
"""
Enhanced utility module for generating summaries using local LLMs
Supports Hugging Face transformers
"""

import os
import logging

import json
from datetime import datetime, timedelta
from typing import List, Optional
from db_utils import get_database_manager
from summary_config import SummaryConfig

class LLMTranscriptionSummarizer:
    """Enhanced summarizer with LLM support for better summaries"""
    
    def __init__(self, db_path="transcriptions/transcriptions.db", 
                 summarization_method="simple", model_name=None):
        """
        Initialize the summarizer
        
        Args:
            db_path: Path to SQLite database
            summarization_method: "simple" or "transformers"
            model_name: Model name for transformers (optional)
        """
        self.db_path = db_path
        self.summarization_method = summarization_method
        self.model_name = model_name or self._get_default_model()
        
        self.setup_logging()
        self.setup_database()
        self.setup_summarizer()
    
    def _get_default_model(self):
        """Get default model based on method"""
        if self.summarization_method == "transformers":
            return "sshleifer/distilbart-cnn-12-6"  # Lightweight distilled BART for summarization

        return None
    
    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Setup database using the improved database manager"""
        if not os.path.exists(self.db_path):
            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.db_manager = get_database_manager(self.db_path)
    
    def setup_summarizer(self):
        """Initialize the summarization model"""
        self.summarizer = None
        self.tokenizer = None
        
        if self.summarization_method == "transformers":
            try:
                from transformers import pipeline, AutoTokenizer
                
                self.logger.info(f"Loading transformers model: {self.model_name}")
                
                # Load tokenizer to check input lengths
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load summarization pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self._has_cuda() else -1  # Use GPU if available
                )
                
                self.logger.info("Transformers summarization model loaded successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to load transformers model: {e}")
                self.logger.info("Falling back to simple summarization")
                self.summarization_method = "simple"
        

    
    def _has_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    

    
    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def preprocess_text_with_prompt(self, text: str) -> str:
        """Add custom prompting context to the text using configuration"""
        # Apply custom preprocessing if configured
        try:
            if hasattr(SummaryConfig, 'custom_preprocessor'):
                text = SummaryConfig.custom_preprocessor(text)
        except Exception as e:
            self.logger.warning(f"Custom preprocessor failed: {e}")
        
        # Skip very short transcriptions if configured
        if SummaryConfig.SKIP_VERY_SHORT:
            if len(text.split()) < SummaryConfig.MIN_WORDS_THRESHOLD:
                return text  # Return as-is for very short text
        
        # Get the current style configuration
        style = SummaryConfig.STYLE
        prompt_config = SummaryConfig.PROMPTS.get(style, SummaryConfig.PROMPTS["meeting_focused"])
        
        # Use simple prefix by default, but allow template override
        if "template" in prompt_config and prompt_config["template"].strip():
            # Use full template approach
            return prompt_config["template"].format(text=text)
        else:
            # Use simple prefix approach
            return prompt_config["prefix"] + text

    def summarize_with_transformers(self, text: str) -> str:
        """Generate summary using transformers model"""
        try:
            # Apply custom prompting
            prompted_text = self.preprocess_text_with_prompt(text)
            
            # Check token length and chunk if necessary
            tokens = self.tokenizer.encode(prompted_text, truncation=False)
            max_tokens = self.tokenizer.model_max_length - 100  # Leave room for output
            
            if len(tokens) > max_tokens:
                # Split into chunks
                chunks = self.chunk_text(text, max_chunk_size=800)
                summaries = []
                
                for chunk in chunks:
                    # Apply prompting to each chunk
                    prompted_chunk = self.preprocess_text_with_prompt(chunk)
                    chunk_summary = self.summarizer(
                        prompted_chunk,
                        max_length=130,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
                
                # If we have multiple chunk summaries, summarize them
                if len(summaries) > 1:
                    combined = " ".join(summaries)
                    if len(self.tokenizer.encode(combined)) <= max_tokens:
                        final_summary = self.summarizer(
                            combined,
                            max_length=200,
                            min_length=50,
                            do_sample=False
                        )[0]['summary_text']
                    else:
                        # Just take the first few summaries
                        final_summary = " ".join(summaries[:2])
                else:
                    final_summary = summaries[0]
            else:
                # Single pass summarization with configurable compression
                input_length = len(self.tokenizer.encode(text))
                
                # Use configuration settings for compression ratios
                target_compression = getattr(SummaryConfig, 'TARGET_COMPRESSION', 0.4)
                min_compression = getattr(SummaryConfig, 'MIN_COMPRESSION', 0.15)
                
                max_len = max(30, min(100, int(input_length * target_compression)))
                min_len = max(10, min(30, int(input_length * min_compression)))
                
                final_summary = self.summarizer(
                    prompted_text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False
                )[0]['summary_text']
            
            # Apply custom post-processing if configured
            try:
                if hasattr(SummaryConfig, 'custom_postprocessor'):
                    final_summary = SummaryConfig.custom_postprocessor(final_summary)
            except Exception as e:
                self.logger.warning(f"Custom postprocessor failed: {e}")
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Transformers summarization failed: {e}")
            return self.simple_summarize(text)
    

    
    def simple_summarize(self, text: str) -> str:
        """Fallback simple extractive summarization"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text.strip()
        
        word_count = len(text.split())
        
        if word_count < 50:
            return text.strip()
        elif word_count < 200:
            summary_sentences = sentences[:2]
            if len(sentences) > 2:
                summary_sentences.append(sentences[-1])
            return ". ".join(summary_sentences) + "."
        else:
            mid_idx = len(sentences) // 2
            summary_sentences = [sentences[0], sentences[mid_idx], sentences[-1]]
            return ". ".join(summary_sentences) + "."
    
    def generate_summary_text(self, transcriptions):
        """Generate a summary from a list of transcriptions using configured method"""
        if not transcriptions:
            return ""
        
        # Combine all text
        combined_text = " ".join([t['text'] for t in transcriptions])
        
        # Choose summarization method
        if self.summarization_method == "transformers" and self.summarizer:
            return self.summarize_with_transformers(combined_text)

        else:
            return self.simple_summarize(combined_text)
    
    # Include all other methods from the original class
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
                current_start_time = current_start_time.replace(
                    minute=(current_start_time.minute // interval_minutes) * interval_minutes
                )
            
            current_end_time = current_start_time + timedelta(minutes=interval_minutes)
            
            if timestamp < current_end_time:
                current_group.append({
                    'timestamp': timestamp,
                    'text': text,
                    'confidence': confidence
                })
            else:
                if current_group:
                    grouped_data.append({
                        'start_time': current_start_time,
                        'end_time': current_end_time,
                        'transcriptions': current_group
                    })
                
                current_start_time = timestamp.replace(second=0, microsecond=0)
                current_start_time = current_start_time.replace(
                    minute=(current_start_time.minute // interval_minutes) * interval_minutes
                )
                current_end_time = current_start_time + timedelta(minutes=interval_minutes)
                current_group = [{
                    'timestamp': timestamp,
                    'text': text,
                    'confidence': confidence
                }]
        
        if current_group:
            grouped_data.append({
                'start_time': current_start_time,
                'end_time': current_end_time,
                'transcriptions': current_group
            })
        
        return grouped_data
    
    def save_summary(self, start_time, end_time, transcriptions):
        """Save a summary to the summaries table with method tracking"""
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
                self.summarization_method, self.model_name
            )
            self.logger.info(f"Saved {self.summarization_method} summary for {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
    
    def generate_daily_summaries(self, date=None, interval_minutes=10):
        """Generate summaries for a specific date grouped by intervals"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Generating {interval_minutes}-minute summaries for {date} using {self.summarization_method}")
        
        grouped_data = self.group_transcriptions_by_interval(interval_minutes, date)
        
        if not grouped_data:
            self.logger.info(f"No transcriptions found for {date}")
            return 0
        
        summaries_created = 0
        
        for group in grouped_data:
            start_time = group['start_time']
            end_time = group['end_time']
            transcriptions = group['transcriptions']
            
            # Check if summary already exists for this interval and method
            try:
                if not self.db_manager.check_summary_exists(start_time, end_time, self.summarization_method):
                    self.save_summary(start_time, end_time, transcriptions)
                    summaries_created += 1
                else:
                    self.logger.debug(f"Summary already exists for {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
            except Exception as e:
                self.logger.error(f"Failed to process summary for {start_time.strftime('%H:%M')}: {e}")
        
        self.logger.info(f"Created {summaries_created} new summaries for {date}")
        return summaries_created
    
    def get_available_dates(self):
        """Get all dates that have transcriptions"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT date(timestamp) as day
                    FROM transcriptions 
                    ORDER BY day DESC
                ''')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get available dates: {e}")
            return []
    
    def view_summaries(self, date=None, limit=None, method=None):
        """View existing summaries with optional method filter"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT start_timestamp, end_timestamp, summary, word_count, 
                           transcription_count, avg_confidence, created_at, method, model_name
                    FROM summaries
                '''
                params = []
                
                conditions = []
                if date:
                    conditions.append("date(start_timestamp) = ?")
                    params.append(date)
                
                if method:
                    conditions.append("method = ?")
                    params.append(method)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY start_timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Failed to view summaries: {e}")
            return []
    
    def delete_summaries(self, date=None, method=None):
        """Delete summaries with optional filters"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "DELETE FROM summaries"
                count_query = "SELECT COUNT(*) FROM summaries"
                params = []
                
                conditions = []
                if date:
                    conditions.append("date(start_timestamp) = ?")
                    params.append(date)
                
                if method:
                    conditions.append("method = ?")
                    params.append(method)
                
                if conditions:
                    condition_str = " WHERE " + " AND ".join(conditions)
                    query += condition_str
                    count_query += condition_str
                
                cursor.execute(count_query, params)
                count = cursor.fetchone()[0]
                
                cursor.execute(query, params)
                conn.commit()
                
                return count
        except Exception as e:
            self.logger.error(f"Failed to delete summaries: {e}")
            return 0
