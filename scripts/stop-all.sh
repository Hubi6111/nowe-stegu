#!/usr/bin/env bash
# ════════════════════════════════════════════════════════
#  Stegu Visualizer — Stop all services (macOS)
# ════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "═══ Stopping Stegu Visualizer ═══"

for svc in inference api frontend; do
  pidfile="$PROJECT_ROOT/.pid-$svc"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "  ✓ Stopped $svc (PID $pid)"
    else
      echo "  · $svc already stopped"
    fi
    rm -f "$pidfile"
  else
    echo "  · $svc — no PID file"
  fi
done

# Also kill any lingering processes on the service ports
for port in 3000 8000 8001; do
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
    echo "  ✓ Freed port $port"
  fi
done

echo ""
echo "Done. All services stopped."
