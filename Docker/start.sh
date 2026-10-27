#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
docker info --format '{{.ServerVersion}}'
docker compose version
docker compose -f compose.yaml up -d --build --wait --wait-timeout 240 || {
  docker compose -f compose.yaml logs --tail 60
  exit 1
}
printf '\nRAG is ready. Default URL: http://localhost:8501\n'
