#!/usr/bin/env python3
"""
LLM-enhanced script to generate summaries from transcriptions data
Supports multiple local LLM backends: transformers, ollama, or simple fallback
"""

import argparse
import sys
import os
from datetime import datetime

# Import the LLM-enhanced summarizer
from summary_utils_llm import LLMTranscriptionSummarizer

def display_summaries(summaries):
    """Display summaries in a formatted table with method info"""
    if not summaries:
        print("No summaries found")
        return
    
    print(f"\n{'Time Range':<20} {'Method':<12} {'Words':<8} {'Transcripts':<10} {'Confidence':<12} {'Summary':<60}")
    print("-" * 130)
    
    for row in summaries:
        start_time = datetime.fromisoformat(row[0]).strftime("%H:%M")
        end_time = datetime.fromisoformat(row[1]).strftime("%H:%M")
        time_range = f"{start_time}-{end_time}"
        
        summary_preview = row[2][:57] + "..." if len(row[2]) > 60 else row[2]
        method = row[7] if len(row) > 7 else "unknown"
        
        print(f"{time_range:<20} {method:<12} {row[3]:<8} {row[4]:<10} {row[5]:<12.3f} {summary_preview}")

def check_dependencies(method):
    """Check if required dependencies are available"""
    if method == "transformers":
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    elif method == "ollama":
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate LLM-enhanced summaries from transcriptions")
    parser.add_argument("--date", help="Date to generate summaries for (YYYY-MM-DD). Default: today")
    parser.add_argument("--interval", type=int, default=10, help="Summary interval in minutes (default: 10)")
    parser.add_argument("--db", default="transcriptions/transcriptions.db", 
                       help="Path to database (default: transcriptions/transcriptions.db)")
    parser.add_argument("--method", choices=["simple", "transformers", "ollama"], 
                       default="transformers", help="Summarization method (default: transformers)")
    parser.add_argument("--model", help="Model name (e.g., 'facebook/bart-large-cnn' or 'llama3.2:1b')")
    parser.add_argument("--view", action="store_true", help="View existing summaries instead of generating")
    parser.add_argument("--delete", action="store_true", help="Delete summaries for the specified date/method")
    parser.add_argument("--all-dates", action="store_true", help="Generate summaries for all dates with transcriptions")
    parser.add_argument("--filter-method", help="Filter summaries by method when viewing")
    
    args = parser.parse_args()
    
    # Set default date to today if not specified
    if args.date is None and not args.all_dates:
        args.date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Error: Database file {args.db} not found")
        print("Make sure you have run the transcription system first to create the database")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies(args.method):
        if args.method == "transformers":
            print("Warning: transformers or torch not available. Install with: pip install transformers torch")
            print("Falling back to simple method")
            args.method = "simple"
        elif args.method == "ollama":
            print("Warning: Ollama not available. Make sure Ollama is running on localhost:11434")
            print("Falling back to simple method")
            args.method = "simple"
    
    try:
        # Create LLM-enhanced summarizer instance
        summarizer = LLMTranscriptionSummarizer(
            db_path=args.db,
            summarization_method=args.method,
            model_name=args.model
        )
        
        if args.view:
            summaries = summarizer.view_summaries(
                date=args.date, 
                limit=20 if not args.date else None,
                method=args.filter_method
            )
            display_summaries(summaries)
            
            if not args.date and not args.filter_method:
                print("\nTip: Use --filter-method to view summaries by specific method")
            return
        
        if args.delete:
            if args.date or args.filter_method:
                count = summarizer.delete_summaries(date=args.date, method=args.filter_method)
                filter_desc = f" for {args.date}" if args.date else ""
                filter_desc += f" using {args.filter_method}" if args.filter_method else ""
                
                if count > 0:
                    response = input(f"Delete {count} summaries{filter_desc}? (y/N): ")
                    if response.lower() == 'y':
                        summarizer.delete_summaries(date=args.date, method=args.filter_method)
                        print(f"Deleted {count} summaries{filter_desc}")
                    else:
                        print("Operation cancelled")
                else:
                    print(f"No summaries found{filter_desc}")
            else:
                summaries = summarizer.view_summaries()
                count = len(summaries)
                if count > 0:
                    response = input(f"Delete ALL {count} summaries? (y/N): ")
                    if response.lower() == 'y':
                        summarizer.delete_summaries()
                        print(f"Deleted all {count} summaries")
                    else:
                        print("Operation cancelled")
                else:
                    print("No summaries found")
            return
        
        if args.all_dates:
            # Get all dates with transcriptions
            dates = summarizer.get_available_dates()
            
            if not dates:
                print("No transcriptions found in database")
                return
            
            print(f"Found transcriptions for {len(dates)} dates: {', '.join(dates)}")
            print(f"Using {args.method} method" + (f" with model {args.model}" if args.model else ""))
            
            total_summaries = 0
            for date in dates:
                print(f"\nProcessing {date}...")
                summaries_created = summarizer.generate_daily_summaries(date, args.interval)
                total_summaries += summaries_created
            
            print(f"\nTotal summaries created: {total_summaries}")
            
        else:
            # Generate summaries for specific date
            print(f"Generating {args.interval}-minute summaries for {args.date}")
            print(f"Using {args.method} method" + (f" with model {args.model}" if args.model else ""))
            
            summaries_created = summarizer.generate_daily_summaries(args.date, args.interval)
            
            if summaries_created > 0:
                print(f"\nSuccess! Created {summaries_created} summaries using {args.method}")
                print(f"Use --view option to see the generated summaries")
            else:
                print("No new summaries were created (they may already exist or no transcriptions found)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
