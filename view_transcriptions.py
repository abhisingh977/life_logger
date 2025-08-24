#!/usr/bin/env python3
"""
Script to view transcriptions from the SQLite database
"""

import sqlite3
import argparse
from datetime import datetime, timedelta
import pandas as pd

def connect_db(db_path):
    """Connect to the SQLite database"""
    return sqlite3.connect(db_path)

def view_all_transcriptions(db_path):
    """View all transcriptions"""
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, text, confidence, duration 
        FROM transcriptions 
        ORDER BY timestamp DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    if results:
        print(f"\n{'ID':<5} {'Timestamp':<20} {'Confidence':<12} {'Duration':<10} {'Text':<50}")
        print("-" * 100)
        for row in results:
            print(f"{row[0]:<5} {row[1]:<20} {row[2]:<12.3f} {row[3]:<10.2f} {row[4][:47]+'...' if len(row[4]) > 50 else row[4]}")
    else:
        print("No transcriptions found.")

def view_by_date(db_path, date_str):
    """View transcriptions for a specific date"""
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, text, confidence, duration 
        FROM transcriptions 
        WHERE date(timestamp) = ?
        ORDER BY timestamp
    ''', (date_str,))
    
    results = cursor.fetchall()
    conn.close()
    
    if results:
        print(f"\nTranscriptions for {date_str}:")
        print(f"{'Time':<10} {'Confidence':<12} {'Text':<50}")
        print("-" * 75)
        for row in results:
            time_only = row[1].split('T')[1][:8] if 'T' in row[1] else row[1][-8:]
            print(f"{time_only:<10} {row[3]:<12.3f} {row[2]}")
    else:
        print(f"No transcriptions found for {date_str}")

def view_summary(db_path):
    """View summary statistics"""
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    # Total count
    cursor.execute('SELECT COUNT(*) FROM transcriptions')
    total_count = cursor.fetchone()[0]
    
    # Count by date
    cursor.execute('''
        SELECT date(timestamp) as day, COUNT(*) as count 
        FROM transcriptions 
        GROUP BY date(timestamp)
        ORDER BY day DESC
    ''')
    daily_counts = cursor.fetchall()
    
    # Average confidence
    cursor.execute('SELECT AVG(confidence) FROM transcriptions')
    avg_confidence = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n=== TRANSCRIPTION SUMMARY ===")
    print(f"Total transcriptions: {total_count}")
    print(f"Average confidence: {avg_confidence:.3f}" if avg_confidence else "Average confidence: N/A")
    print(f"\nDaily breakdown:")
    print(f"{'Date':<12} {'Count':<6}")
    print("-" * 20)
    for day, count in daily_counts:
        print(f"{day:<12} {count:<6}")

def export_to_csv(db_path, output_file="transcriptions_export.csv"):
    """Export all transcriptions to CSV"""
    conn = connect_db(db_path)
    
    df = pd.read_sql_query('''
        SELECT id, timestamp, text, confidence, duration, audio_file
        FROM transcriptions
        ORDER BY timestamp
    ''', conn)
    
    conn.close()
    
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} transcriptions to {output_file}")

def search_transcriptions(db_path, search_term):
    """Search for specific text in transcriptions"""
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, text, confidence 
        FROM transcriptions 
        WHERE text LIKE ?
        ORDER BY timestamp DESC
    ''', (f'%{search_term}%',))
    
    results = cursor.fetchall()
    conn.close()
    
    if results:
        print(f"\nSearch results for '{search_term}':")
        print(f"{'ID':<5} {'Timestamp':<20} {'Confidence':<12} {'Text':<50}")
        print("-" * 90)
        for row in results:
            # Highlight search term in text
            highlighted_text = row[2].replace(search_term, f"**{search_term}**")
            print(f"{row[0]:<5} {row[1]:<20} {row[3]:<12.3f} {highlighted_text}")
    else:
        print(f"No transcriptions found containing '{search_term}'")

def view_summaries(db_path, date=None):
    """View summaries"""
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    if date:
        cursor.execute('''
            SELECT start_timestamp, end_timestamp, summary, word_count, 
                   transcription_count, avg_confidence, created_at
            FROM summaries 
            WHERE date(start_timestamp) = ?
            ORDER BY start_timestamp
        ''', (date,))
    else:
        cursor.execute('''
            SELECT start_timestamp, end_timestamp, summary, word_count, 
                   transcription_count, avg_confidence, created_at
            FROM summaries 
            ORDER BY start_timestamp DESC
            LIMIT 20
        ''')
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print(f"No summaries found" + (f" for {date}" if date else ""))
        return
    
    print(f"\n{'Time Range':<20} {'Words':<8} {'Transcripts':<12} {'Confidence':<12} {'Summary':<60}")
    print("-" * 120)
    
    for row in results:
        start_time = datetime.fromisoformat(row[0]).strftime("%H:%M")
        end_time = datetime.fromisoformat(row[1]).strftime("%H:%M")
        time_range = f"{start_time}-{end_time}"
        
        summary_preview = row[2][:57] + "..." if len(row[2]) > 60 else row[2]
        
        print(f"{time_range:<20} {row[3]:<8} {row[4]:<12} {row[5]:<12.3f} {summary_preview}")

def main():
    parser = argparse.ArgumentParser(description="View transcriptions from SQLite database")
    parser.add_argument("--db", default="/home/abhishek/abhi/logging/transcriptions/transcriptions.db",
                       help="Path to the SQLite database")
    parser.add_argument("--all", action="store_true", help="View all transcriptions")
    parser.add_argument("--date", help="View transcriptions for specific date (YYYY-MM-DD)")
    parser.add_argument("--today", action="store_true", help="View today's transcriptions")
    parser.add_argument("--summary", action="store_true", help="View summary statistics")
    parser.add_argument("--summaries", action="store_true", help="View generated summaries")
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--search", help="Search for text in transcriptions")
    
    args = parser.parse_args()
    
    if not any([args.all, args.date, args.today, args.summary, args.summaries, args.export, args.search]):
        # Default behavior: show summary
        view_summary(args.db)
        return
    
    if args.all:
        view_all_transcriptions(args.db)
    
    if args.date:
        if args.summaries:
            view_summaries(args.db, args.date)
        else:
            view_by_date(args.db, args.date)
    
    if args.today:
        today = datetime.now().strftime("%Y-%m-%d")
        if args.summaries:
            view_summaries(args.db, today)
        else:
            view_by_date(args.db, today)
    
    if args.summary:
        view_summary(args.db)
    
    if args.summaries and not args.date and not args.today:
        view_summaries(args.db)
    
    if args.export:
        export_to_csv(args.db, args.export)
    
    if args.search:
        search_transcriptions(args.db, args.search)

if __name__ == "__main__":
    main()
