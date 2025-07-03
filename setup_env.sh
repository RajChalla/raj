#!/usr/bin/env bash
# Install project dependencies with optional proxy support.
# Usage:
#   ./setup_env.sh [http://proxy:port]
# If a proxy URL is provided, it will be used for pip.

PROXY_URL="$1"

if [ -n "$PROXY_URL" ]; then
    pip install --proxy "$PROXY_URL" -r requirements.txt
else
    pip install -r requirements.txt
fi
