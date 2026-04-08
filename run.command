#!/usr/bin/env bash
# ════════════════════════════════════════════════════════
#  Stegu Visualizer — macOS double-click launcher
#
#  Double-click this file in Finder to start the full stack:
#  inference :8001 + API :8000 + Next :3000 (same as: npm run dev)
#  Open http://localhost:3000 in your browser.
# ════════════════════════════════════════════════════════

# Move to the directory where this file lives (project root)
cd "$(dirname "$0")"

exec bash scripts/start-all.sh
