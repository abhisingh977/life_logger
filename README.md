# Life Logger

A real-time audio transcription and life logging system using Whisper AI and LLM summarization.

## Features

- Real-time audio transcription using OpenAI Whisper
- Continuous daemon mode with automatic periodic summarization
- LLM-powered intelligent summarization using transformers
- SQLite database for persistent storage
- GUI viewer for browsing transcriptions
- Multi-language support with optional translation

## Quick Start

1. **Setup**:

   ```bash
   ./run.sh setup
   ```

2. **Start daemon mode** (continuous transcription + periodic summaries):

   ```bash
   ./run.sh daemon start
   ```

3. **View transcriptions**:

   ```bash
   ./run.sh view --last 24
   ./run.sh gui  # Launch GUI viewer
   ```

4. **Stop daemon**:
   ```bash
   ./run.sh daemon stop
   ```

## Usage

Run `./run.sh help` for complete usage instructions and examples.

## Requirements

- Python 3.8+
- Poetry (for dependency management)
- Microphone access
- Optional: PyTorch and transformers for LLM summarization

## Project Structure

- `transcript.py` - Core real-time transcription engine
- `scheduler.py` - Daemon mode scheduler
- `generate_summaries_llm.py` - LLM-powered summarization
- `view_transcriptions.py` - CLI transcription viewer
- `transcription_viewer_gui.py` - GUI transcription viewer
- `run.sh` - Main runner script with all commands
- `transcriptions/` - Database and audio chunk storage
