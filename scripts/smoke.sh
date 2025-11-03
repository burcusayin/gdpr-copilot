#!/usr/bin/env bash
set -euo pipefail

echo "[healthz]"
curl -s http://localhost:8080/healthz
echo

echo "[query]"
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the data processing principles?"}'
echo