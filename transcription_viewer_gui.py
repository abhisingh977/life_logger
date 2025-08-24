#!/usr/bin/env python3
"""
Simple GUI to view transcriptions from the SQLite database
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
from datetime import datetime, timedelta
import os

class TranscriptionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcription Viewer")
        self.root.geometry("1000x600")
        
        self.db_path = "/home/abhishek/abhi/logging/transcriptions/transcriptions.db"
        
        self.create_widgets()
        self.load_data()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Date selection
        ttk.Label(control_frame, text="Date:").grid(row=0, column=0, padx=(0, 5))
        
        self.date_var = tk.StringVar(value="All")
        self.date_combo = ttk.Combobox(control_frame, textvariable=self.date_var, width=15)
        self.date_combo.grid(row=0, column=1, padx=(0, 10))
        self.date_combo.bind("<<ComboboxSelected>>", self.on_date_change)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=self.load_data).grid(row=0, column=2, padx=(0, 5))
        
        # Export button
        ttk.Button(control_frame, text="Export CSV", command=self.export_csv).grid(row=0, column=3, padx=(0, 5))
        
        # Search frame
        search_frame = ttk.Frame(control_frame)
        search_frame.grid(row=0, column=4, padx=(10, 0))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(5, 0))
        search_entry.bind("<Return>", lambda e: self.search_transcriptions())
        
        ttk.Button(search_frame, text="Search", command=self.search_transcriptions).pack(side=tk.LEFT, padx=(5, 0))
        
        # Treeview for displaying transcriptions
        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        columns = ("ID", "Timestamp", "Text", "Confidence", "Duration")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        self.tree.heading("ID", text="ID")
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.heading("Text", text="Text")
        self.tree.heading("Confidence", text="Confidence")
        self.tree.heading("Duration", text="Duration")
        
        # Set column widths
        self.tree.column("ID", width=50)
        self.tree.column("Timestamp", width=150)
        self.tree.column("Text", width=500)
        self.tree.column("Confidence", width=100)
        self.tree.column("Duration", width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Grid treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Bind double-click to view full text
        self.tree.bind("<Double-1>", self.view_full_text)
    
    def connect_db(self):
        """Connect to the SQLite database"""
        if not os.path.exists(self.db_path):
            messagebox.showerror("Error", f"Database not found: {self.db_path}")
            return None
        return sqlite3.connect(self.db_path)
    
    def load_data(self):
        """Load data from database"""
        conn = self.connect_db()
        if not conn:
            return
        
        cursor = conn.cursor()
        
        # Load available dates for dropdown
        cursor.execute('''
            SELECT DISTINCT date(timestamp) as day 
            FROM transcriptions 
            ORDER BY day DESC
        ''')
        dates = ["All"] + [row[0] for row in cursor.fetchall()]
        self.date_combo['values'] = dates
        
        # Load transcriptions based on selected date
        selected_date = self.date_var.get()
        if selected_date == "All" or not selected_date:
            cursor.execute('''
                SELECT id, timestamp, text, confidence, duration 
                FROM transcriptions 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
        else:
            cursor.execute('''
                SELECT id, timestamp, text, confidence, duration 
                FROM transcriptions 
                WHERE date(timestamp) = ?
                ORDER BY timestamp DESC
            ''', (selected_date,))
        
        transcriptions = cursor.fetchall()
        conn.close()
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Insert new data
        for transcription in transcriptions:
            # Format timestamp for display
            timestamp_str = transcription[1]
            if 'T' in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str)
                display_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                display_timestamp = timestamp_str
            
            # Truncate text for display
            text = transcription[2]
            display_text = (text[:100] + "...") if len(text) > 100 else text
            
            self.tree.insert("", "end", values=(
                transcription[0],  # ID
                display_timestamp,
                display_text,
                f"{transcription[3]:.3f}" if transcription[3] is not None else "N/A",
                f"{transcription[4]:.2f}s" if transcription[4] is not None else "N/A"
            ))
        
        self.status_var.set(f"Loaded {len(transcriptions)} transcriptions")
    
    def on_date_change(self, event=None):
        """Handle date selection change"""
        self.load_data()
    
    def search_transcriptions(self):
        """Search for transcriptions containing specific text"""
        search_term = self.search_var.get().strip()
        if not search_term:
            self.load_data()
            return
        
        conn = self.connect_db()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, text, confidence, duration 
            FROM transcriptions 
            WHERE text LIKE ?
            ORDER BY timestamp DESC
        ''', (f'%{search_term}%',))
        
        transcriptions = cursor.fetchall()
        conn.close()
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Insert search results
        for transcription in transcriptions:
            # Format timestamp for display
            timestamp_str = transcription[1]
            if 'T' in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str)
                display_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                display_timestamp = timestamp_str
            
            # Highlight search term in text
            text = transcription[2]
            display_text = (text[:100] + "...") if len(text) > 100 else text
            
            self.tree.insert("", "end", values=(
                transcription[0],  # ID
                display_timestamp,
                display_text,
                f"{transcription[3]:.3f}" if transcription[3] is not None else "N/A",
                f"{transcription[4]:.2f}s" if transcription[4] is not None else "N/A"
            ))
        
        self.status_var.set(f"Found {len(transcriptions)} transcriptions containing '{search_term}'")
    
    def view_full_text(self, event):
        """View full text of selected transcription"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        transcription_id = item['values'][0]
        
        conn = self.connect_db()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, text FROM transcriptions WHERE id = ?', (transcription_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Create popup window for full text
            popup = tk.Toplevel(self.root)
            popup.title(f"Transcription #{transcription_id}")
            popup.geometry("600x400")
            
            # Add timestamp label
            timestamp_label = ttk.Label(popup, text=f"Timestamp: {result[0]}", font=("TkDefaultFont", 10, "bold"))
            timestamp_label.pack(pady=10)
            
            # Add text widget with scrollbar
            text_frame = ttk.Frame(popup)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("TkDefaultFont", 11))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            text_widget.insert(tk.END, result[1])
            text_widget.config(state=tk.DISABLED)
            
            # Close button
            ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
    
    def export_csv(self):
        """Export current view to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save transcriptions as CSV"
        )
        
        if filename:
            try:
                conn = self.connect_db()
                if not conn:
                    return
                
                import csv
                cursor = conn.cursor()
                
                # Get current filter
                selected_date = self.date_var.get()
                if selected_date == "All" or not selected_date:
                    cursor.execute('''
                        SELECT id, timestamp, text, confidence, duration, audio_file
                        FROM transcriptions 
                        ORDER BY timestamp DESC
                    ''')
                else:
                    cursor.execute('''
                        SELECT id, timestamp, text, confidence, duration, audio_file
                        FROM transcriptions 
                        WHERE date(timestamp) = ?
                        ORDER BY timestamp DESC
                    ''', (selected_date,))
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['ID', 'Timestamp', 'Text', 'Confidence', 'Duration', 'Audio File'])
                    writer.writerows(cursor.fetchall())
                
                conn.close()
                messagebox.showinfo("Success", f"Exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")

def main():
    root = tk.Tk()
    app = TranscriptionViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
