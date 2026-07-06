#!/usr/bin/env sh
set -eu

suite="${GW_PERF_SUITE:-frontier}"
timeout="${GW_PERF_TIMEOUT:-75}"
frontier_seconds="${GW_PERF_FRONTIER_SECONDS:-55}"
repeat="${GW_PERF_REPEAT:-1}"
graph_cache_mode="${GW_PERF_GRAPH_CACHE_MODE:-shared}"

exec scripts/perf_frontiers.py \
  --suite "$suite" \
  --timeout "$timeout" \
  --frontier-seconds "$frontier_seconds" \
  --repeat "$repeat" \
  --graph-cache-mode "$graph_cache_mode" \
  "$@"
