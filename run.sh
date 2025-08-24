#!/bin/bash

# Logging Transcription System Runner
# Uses Poetry for environment management

set -e  # Exit on any error

PROJECT_DIR="/home/abhishek/abhi/life_logger"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if Poetry is installed
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        error "Poetry is not installed. Please install it first:"
        echo "curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    log "Installing dependencies with Poetry..."
    poetry install
    log "Dependencies installed successfully!"
}

# Setup function
setup() {
    log "Setting up the logging transcription system..."
    
    check_poetry
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log "Creating virtual environment..."
        poetry env use python3
    fi
    
    # Install dependencies
    install_deps
    
    # Create necessary directories
    mkdir -p transcriptions
    
    log "Setup completed successfully!"
}

# Show help
show_help() {
    echo -e "${BLUE}Logging Transcription System${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                   - Setup Poetry environment and dependencies"
    echo "  transcript [OPTIONS]    - Start real-time transcription (single run)"
    echo "  daemon start [INTERVAL] [LANG] [--translate] - Start continuous transcription + periodic summaries"
    echo "  daemon stop             - Stop daemon mode"
    echo "  daemon restart [INT] [LANG] - Restart daemon with optional interval and language"
    echo "  daemon status           - Show daemon status"
    echo "  summarize [OPTIONS]     - Generate summaries from transcriptions"
    echo "  view [OPTIONS]          - View stored transcriptions"
    echo "  gui                     - Launch GUI viewer"
    echo "  shell                   - Enter Poetry shell"
    echo "  clean                   - Clean up generated files"
    echo "  help                    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                               # Initial setup"
    echo "  $0 daemon start                        # Start continuous mode (20min summaries)"
    echo "  $0 daemon start 30                     # Start with 30-minute summary intervals"
    echo "  $0 daemon start 20 hi                  # Start with Hindi transcription"
    echo "  $0 daemon start 10 en                  # Start with English transcription"
    echo "  $0 daemon start 10 --translate         # Auto-translate to English"
    echo "  $0 daemon start 5 hi --translate       # Hindi with translation to English"
    echo "  $0 daemon status                       # Check if running"
    echo "  $0 daemon stop                         # Stop continuous mode"
    echo "  $0 transcript --continuous             # Manual continuous transcription"
    echo "  $0 transcript --model small            # Single-run with small model"
    echo "  $0 transcript --language hi           # Transcribe in Hindi"
    echo "  $0 transcript --translate              # Auto-translate to English"
    echo "  $0 summarize --method transformers     # Generate summaries with transformers"
    echo "  $0 view --last 24                     # View last 24 hours"
    echo ""
    echo "Daemon Mode:"
    echo "  The daemon runs continuous transcription and periodic summarization."
    echo "  Default summary interval is 20 minutes, customizable with:"
    echo "    $0 daemon start [MINUTES] [LANGUAGE]"
    echo "  Language codes: hi (Hindi), en (English), auto-detect if not specified"
    echo "  Logs are written to: scheduler.log, transcription.log"
    echo ""
}

# PID files
SCHEDULER_PID_FILE="scheduler.pid"
TRANSCRIPT_PID_FILE="transcript.pid"

# Run transcription
run_transcript() {
    log "Starting real-time transcription..."
    poetry run python transcript.py "$@"
}

# Run transcription in daemon mode
start_daemon() {
    local interval=${1:-20}
    local language=""
    local translate=false
    
    # Parse arguments (can be in any order)
    shift  # Remove interval
    while [[ $# -gt 0 ]]; do
        case $1 in
            --translate)
                translate=true
                shift
                ;;
            *)
                if [[ -z "$language" ]]; then
                    language="$1"
                fi
                shift
                ;;
        esac
    done
    
    if is_daemon_running; then
        warn "Daemon is already running (PID: $(cat $SCHEDULER_PID_FILE))"
        return 0
    fi
    
    # Build status message
    local status_msg="Starting transcription daemon with ${interval}-minute summary intervals"
    if [ -n "$language" ]; then
        status_msg="$status_msg (Language: $language"
    else
        status_msg="$status_msg (Multilingual auto-detection"
    fi
    if [ "$translate" = true ]; then
        status_msg="$status_msg, Auto-translate to English"
    fi
    status_msg="$status_msg)..."
    log "$status_msg"
    
    # Start the scheduler in background
    local cmd="nohup poetry run python scheduler.py --interval \"$interval\" --project-dir \"$PROJECT_DIR\""
    if [ -n "$language" ]; then
        cmd="$cmd --language \"$language\""
    fi
    if [ "$translate" = true ]; then
        cmd="$cmd --translate"
    fi
    cmd="$cmd > scheduler_output.log 2>&1 &"
    eval $cmd
    
    local scheduler_pid=$!
    echo $scheduler_pid > "$SCHEDULER_PID_FILE"
    
    # Wait a moment to see if it started successfully
    sleep 3
    
    if is_daemon_running; then
        log "✓ Daemon started successfully (PID: $scheduler_pid)"
        log "  - Continuous transcription: RUNNING"
        log "  - Summary interval: ${interval} minutes"
        if [ -n "$language" ]; then
            log "  - Language: $language"
        else
            log "  - Language: Multilingual (auto-detection)"
        fi
        if [ "$translate" = true ]; then
            log "  - Translation: Auto-translate to English ✓"
        else
            log "  - Translation: Disabled"
        fi
        log "  - Logs: scheduler.log, transcription.log"
        log "  - Status: ./run.sh daemon status"
        log "  - Stop: ./run.sh daemon stop"
    else
        error "✗ Failed to start daemon"
        rm -f "$SCHEDULER_PID_FILE"
        return 1
    fi
}

# Stop daemon
stop_daemon() {
    if ! is_daemon_running; then
        warn "Daemon is not running"
        return 0
    fi
    
    local pid=$(cat "$SCHEDULER_PID_FILE")
    log "Stopping daemon (PID: $pid)..."
    
    # Send TERM signal
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if kill -0 "$pid" 2>/dev/null; then
            warn "Daemon didn't stop gracefully, forcing shutdown..."
            kill -KILL "$pid" 2>/dev/null
        fi
    fi
    
    rm -f "$SCHEDULER_PID_FILE"
    log "✓ Daemon stopped"
}

# Restart daemon
restart_daemon() {
    log "Restarting daemon..."
    stop_daemon
    sleep 2
    start_daemon "$@"  # Pass all arguments to start_daemon
}

# Check if daemon is running
is_daemon_running() {
    [ -f "$SCHEDULER_PID_FILE" ] && kill -0 "$(cat "$SCHEDULER_PID_FILE")" 2>/dev/null
}

# Get daemon status
daemon_status() {
    info "Daemon Status Check"
    echo "=================="
    
    if is_daemon_running; then
        local pid=$(cat "$SCHEDULER_PID_FILE")
        log "✓ Scheduler running (PID: $pid)"
        
        # Check transcription process
        if pgrep -f "transcript.py.*--continuous" > /dev/null; then
            log "✓ Transcription process running"
        else
            warn "⚠ Transcription process not found"
        fi
        
        # Check recent activity
        if [ -f "transcription.log" ]; then
            local last_log=$(tail -1 transcription.log 2>/dev/null | cut -d' ' -f1-2 2>/dev/null)
            info "Last transcription activity: ${last_log:-Unknown}"
        fi
        
        # Check scheduler logs
        if [ -f "scheduler.log" ]; then
            local last_scheduler=$(tail -1 scheduler.log 2>/dev/null | cut -d' ' -f1-2 2>/dev/null)
            info "Last scheduler activity: ${last_scheduler:-Unknown}"
        fi
        
    else
        warn "✗ Daemon not running"
        
        # Check for stale PID file
        if [ -f "$SCHEDULER_PID_FILE" ]; then
            warn "Found stale PID file, removing..."
            rm -f "$SCHEDULER_PID_FILE"
        fi
    fi
    
    # Show process information
    echo ""
    info "Related processes:"
    pgrep -af "transcript.py\|scheduler.py\|generate_summaries" || echo "  None found"
}

# Run summarization
run_summarize() {
    log "Generating summaries..."
    poetry run python generate_summaries_llm.py "$@"
}

# View transcriptions
run_view() {
    log "Viewing transcriptions..."
    poetry run python view_transcriptions.py "$@"
}

# Run GUI
run_gui() {
    log "Starting GUI viewer..."
    poetry run python transcription_viewer_gui.py "$@"
}



# Enter Poetry shell
enter_shell() {
    log "Entering Poetry shell..."
    poetry shell
}

# Clean up generated files
clean() {
    warn "Cleaning up generated files..."
    read -p "This will delete transcription logs and temporary files. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f transcription.log
        # Clean up transcription logs only
        echo "Note: Database and transcription files are preserved"
        log "Cleanup completed!"
    else
        info "Cleanup cancelled."
    fi
}

# Health check
health_check() {
    log "Running health check..."
    
    # Check Python
    if poetry run python --version &> /dev/null; then
        log "✓ Python environment is working"
    else
        error "✗ Python environment issue"
        return 1
    fi
    
    # Check database
    if [ -f "transcriptions/transcriptions.db" ]; then
        log "✓ Database file exists"
    else
        warn "⚠ Database file not found (will be created on first run)"
    fi
    

    
    log "Health check completed!"
}

# Main script logic
main() {
    # Ensure we're in the right directory
    cd "$PROJECT_DIR"
    
    case "${1:-help}" in
        setup)
            setup
            ;;
        transcript)
            shift
            check_poetry
            run_transcript "$@"
            ;;
        daemon)
            shift
            check_poetry
            case "${1:-status}" in
                start)
                    shift
                    start_daemon "$@"
                    ;;
                stop)
                    stop_daemon
                    ;;
                restart)
                    shift
                    restart_daemon "$@"
                    ;;
                status)
                    daemon_status
                    ;;
                *)
                    error "Unknown daemon command: $1"
                    echo "Usage: $0 daemon {start [INTERVAL] [LANG] [--translate]|stop|restart [INTERVAL] [LANG] [--translate]|status}"
                    exit 1
                    ;;
            esac
            ;;
        summarize)
            shift
            check_poetry
            run_summarize "$@"
            ;;
        view)
            shift
            check_poetry
            run_view "$@"
            ;;
        gui)
            shift
            check_poetry
            run_gui "$@"
            ;;

        shell)
            check_poetry
            enter_shell
            ;;
        clean)
            clean
            ;;
        health)
            check_poetry
            health_check
            ;;
        install)
            check_poetry
            install_deps
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
