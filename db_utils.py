#!/usr/bin/env python3
"""
Database utilities with improved SQLite concurrency handling
Fixes database locking issues between transcription and summarization processes
"""

import sqlite3
import time
import logging
import contextlib
from pathlib import Path


class DatabaseManager:
    """Handles SQLite database connections with proper locking and WAL mode"""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.setup_database()
    
    def setup_database(self):
        """Setup database with WAL mode for better concurrency"""
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database with proper settings
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=1000")
            conn.execute("PRAGMA temp_store=memory")
            
            # Create tables
            self.create_tables(conn)
    
    def create_tables(self, conn):
        """Create all necessary tables"""
        cursor = conn.cursor()
        
        # Transcriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                text TEXT,
                confidence REAL,
                duration REAL,
                audio_file TEXT
            )
        ''')
        
        # Enhanced summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_timestamp TEXT,
                end_timestamp TEXT,
                summary TEXT,
                word_count INTEGER,
                transcription_count INTEGER,
                avg_confidence REAL,
                created_at TEXT,
                method TEXT,
                model_name TEXT
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_transcriptions_timestamp 
            ON transcriptions(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_summaries_timestamp 
            ON summaries(start_timestamp, end_timestamp)
        ''')
        
        conn.commit()
    
    @contextlib.contextmanager
    def get_connection(self, timeout=30.0):
        """Get a database connection with retry logic and proper timeout"""
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=timeout,
                    check_same_thread=False
                )
                
                # Set connection pragmas for better concurrency
                conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
                conn.execute("PRAGMA journal_mode=WAL")
                
                try:
                    yield conn
                    return
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Database locked, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Database error after {attempt + 1} attempts: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected database error: {e}")
                raise
    
    def save_transcription(self, transcription):
        """Save transcription to database with retry logic"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check and add new columns for translation support (backward compatibility)
            cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'detected_language' not in columns:
                cursor.execute('ALTER TABLE transcriptions ADD COLUMN detected_language TEXT')
                conn.commit()
            
            if 'original_text' not in columns:
                cursor.execute('ALTER TABLE transcriptions ADD COLUMN original_text TEXT')
                conn.commit()
            
            if 'was_translated' not in columns:
                cursor.execute('ALTER TABLE transcriptions ADD COLUMN was_translated BOOLEAN DEFAULT 0')
                conn.commit()
            
            if 'translation_confidence' not in columns:
                cursor.execute('ALTER TABLE transcriptions ADD COLUMN translation_confidence REAL')
                conn.commit()
            
            cursor.execute('''
                INSERT INTO transcriptions 
                (timestamp, text, confidence, duration, audio_file, detected_language, 
                 original_text, was_translated, translation_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transcription["timestamp"].isoformat(),
                transcription["text"],  # English text (translated if needed)
                transcription["confidence"],
                transcription["duration"],
                transcription["audio_file"],
                transcription.get("detected_language", "unknown"),
                transcription.get("original_text", transcription["text"]),  # Original language text
                transcription.get("was_translated", False),
                transcription.get("translation_confidence")
            ))
            conn.commit()
    
    def save_summary(self, start_time, end_time, summary_text, word_count, 
                    transcription_count, avg_confidence, created_at, method, model_name):
        """Save summary to database with retry logic"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO summaries (start_timestamp, end_timestamp, summary, word_count, 
                                     transcription_count, avg_confidence, created_at, method, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                start_time.isoformat(),
                end_time.isoformat(),
                summary_text,
                word_count,
                transcription_count,
                avg_confidence,
                created_at.isoformat(),
                method,
                model_name
            ))
            conn.commit()
    
    def get_transcriptions_for_date(self, date):
        """Get all transcriptions for a specific date"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, text, confidence 
                FROM transcriptions 
                WHERE date(timestamp) = ?
                ORDER BY timestamp
            ''', (date,))
            return cursor.fetchall()
    
    def check_summary_exists(self, start_time, end_time, method):
        """Check if a summary already exists for the given time range and method"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM summaries 
                WHERE start_timestamp = ? AND end_timestamp = ? AND method = ?
            ''', (start_time.isoformat(), end_time.isoformat(), method))
            return cursor.fetchone() is not None
    
    def get_recent_summaries_count(self, minutes_ago=5, method=None):
        """Get count of summaries created in the last N minutes"""
        from datetime import datetime, timedelta
        
        time_threshold = datetime.now() - timedelta(minutes=minutes_ago)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if method:
                cursor.execute('''
                    SELECT COUNT(*) FROM summaries 
                    WHERE created_at > ? AND method = ?
                ''', (time_threshold.isoformat(), method))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM summaries 
                    WHERE created_at > ?
                ''', (time_threshold.isoformat(),))
            
            return cursor.fetchone()[0]


def get_database_manager(db_path="transcriptions/transcriptions.db"):
    """Get a singleton database manager instance"""
    if not hasattr(get_database_manager, '_instance'):
        get_database_manager._instance = DatabaseManager(db_path)
    return get_database_manager._instance
