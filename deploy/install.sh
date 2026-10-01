#!/bin/sh
# Install/refresh the two launchd services (app on :8501, cloudflared tunnel).
set -e
cd "$(dirname "$0")"
for s in app tunnel; do
  cp com.nyaya.$s.plist ~/Library/LaunchAgents/
  launchctl bootout gui/$(id -u)/com.nyaya.$s 2>/dev/null || true
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nyaya.$s.plist
done
echo "Nyaya service launched on http://127.0.0.1:8501"
