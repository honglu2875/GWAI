#!/usr/bin/env sh
set -eu

profile=debug
if [ "${1:-}" = "--release" ]; then
  profile=release
  shift
fi

if [ "$profile" = "release" ]; then
  cargo build --release --quiet
  target_dir=target/release
else
  cargo build --quiet
  target_dir=target/debug
fi

lib="$target_dir/libgw_pn.rlib"
if [ ! -f "$lib" ]; then
  lib="$(find "$target_dir/deps" -maxdepth 1 -name 'libgw_pn-*.rlib' | head -n 1)"
fi
if [ -z "$lib" ]; then
  echo "error: could not find compiled gw_pn rlib under $target_dir/deps" >&2
  exit 1
fi

out="${TMPDIR:-/tmp}/gw-pn-graph-stats"
rustc scripts/graph-stats.rs \
  --edition=2021 \
  --crate-name graph_stats \
  -L "dependency=$target_dir/deps" \
  --extern "gw_pn=$lib" \
  -o "$out"

"$out" "$@"
