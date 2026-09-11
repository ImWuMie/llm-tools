#!/usr/bin/env bash
set -euo pipefail
HOST="${VLLM_HOST:-127.0.0.1}"
if [[ "$HOST" == "0.0.0.0" ]]; then HOST="127.0.0.1"; fi
PORT="${VLLM_PORT:-8000}"
KEY="${VLLM_API_KEY:-sk-local}"
MODEL="${1:-default}"

curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍 LoRA。\"}]}"
echo
