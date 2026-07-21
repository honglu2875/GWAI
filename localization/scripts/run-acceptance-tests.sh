#!/usr/bin/env sh
set -eu

exec python3 scripts/acceptance_tests.py "$@"
