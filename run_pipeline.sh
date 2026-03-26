#!/bin/bash
# Launch the music processing pipeline inside a tmux session.
# The pipeline keeps running after SSH disconnect.
#
# Usage:
#   ./run_pipeline.sh                       # default settings
#   ./run_pipeline.sh --gemini-concurrency 8 --download-workers 20
#
# Monitor:
#   tmux attach -t pipeline       # live output
#   tail -f pipeline_output.log   # follow stdout
#   tail -f pipeline_errors.log   # follow errors
#   wc -l results.jsonl           # count processed files

set -euo pipefail

# Check if tmux session already exists
if tmux has-session -t pipeline 2>/dev/null; then
    echo "ERROR: tmux session 'pipeline' already exists!"
    echo ""
    echo "  Attach with:  tmux attach -t pipeline"
    echo "  Kill with:    tmux kill-session -t pipeline"
    exit 1
fi

# Start pipeline in a new tmux session, tee output to log file
tmux new-session -d -s pipeline \
    "python3 pipeline.py $* 2>&1 | tee pipeline_output.log; echo ''; echo 'Pipeline exited. Press Enter to close.'; read"

echo "Pipeline started in tmux session 'pipeline'"
echo ""
echo "  Attach:   tmux attach -t pipeline"
echo "  Detach:   Ctrl+B then D"
echo "  Logs:     tail -f pipeline_output.log"
echo "  Errors:   tail -f pipeline_errors.log"
echo "  Progress: wc -l results.jsonl"
