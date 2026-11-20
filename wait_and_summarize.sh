#!/usr/bin/env bash
# Block until one probe has finished, then print its log lines and summary tables.
# Usage: bash wait_and_summarize.sh legibility|handwriting
set -uo pipefail
cd "$(dirname "$0")"
WHAT="$1"
while pgrep -f "^python3 eval_${WHAT}" > /dev/null; do sleep 30; done
echo "=== LOG ==="
if [ "$WHAT" = "handwriting" ]; then
  cat results/hw_*.jsonl > results/handwriting.jsonl
  grep -h -E '^\[(data|skip|done|gen)' logs_hw_*.txt | tail -30
  echo "=== SUMMARY ==="
  python3 summarize.py --legibility none.jsonl 2>&1 | tail -130
else
  grep -E '^\[(skip|done|gen)' logs_legibility.txt | tail -14
  echo "=== SUMMARY ==="
  python3 summarize.py --handwriting none.jsonl 2>&1 | tail -110
fi
