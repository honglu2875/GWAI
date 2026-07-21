#!/usr/bin/env sh
set -eu

# Reuse the frontier runner so hotspot samples retain the same wall-clock,
# baseline, feature, timeout, and graph-cache metadata as broader tuning runs.
GW_PERF_SUITE="${GW_PERF_SUITE:-hotspots}"
GW_PERF_CAPTURE_PROFILE="${GW_PERF_CAPTURE_PROFILE:-1}"
export GW_PERF_SUITE GW_PERF_CAPTURE_PROFILE

exec scripts/run-perf-frontiers.sh "$@"
