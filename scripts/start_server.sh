#!/usr/bin/env bash
# Chạy máy chủ demo textfix nền (nohup), log ở server.log. Dừng: pkill -f server/app.py
#   bash scripts/start_server.sh            # cổng 8088 (hoặc TEXTFIX_PORT trong .env)
#   bash scripts/start_server.sh --model base
set -e
cd "$(dirname "$0")/.."
# image notebook đặt libcuda lệch driver lên đầu LD_LIBRARY_PATH (lỗi 803 ngày 01/10) -> bỏ thư mục compat
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
pkill -f "server/app.py" 2>/dev/null && sleep 2 || true
nohup python server/app.py "$@" > server.log 2>&1 &
echo "đã chạy (pid $!) -- xem log: tail -f server.log ; kiểm tra: curl -s localhost:${TEXTFIX_PORT:-8088}/api/health"
