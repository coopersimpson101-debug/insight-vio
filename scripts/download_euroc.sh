#!/usr/bin/env bash
# Download EuRoC MAV sequences (ASL format zips, ~1-3 GB each) into ./data/euroc/<SEQ>.
# NOT run in CI. The dataset has moved hosts over the years; if the default URL fails,
# get the current link from https://projects.asl.ethz.ch/datasets/ (EuRoC MAV) or the
# ETH Research Collection and set EUROC_BASE_URL accordingly.
#
# usage: scripts/download_euroc.sh [MH_01_easy V1_01_easy ...]
set -euo pipefail
BASE="${EUROC_BASE_URL:-http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset}"
OUT="${EUROC_DIR:-data/euroc}"
SEQS=("$@")
[ ${#SEQS[@]} -eq 0 ] && SEQS=(MH_01_easy V1_01_easy)

folder_for() {
  case "$1" in
    MH_*) echo machine_hall ;;
    V1_*) echo vicon_room1 ;;
    V2_*) echo vicon_room2 ;;
    *) echo "unknown sequence $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$OUT"
for s in "${SEQS[@]}"; do
  url="$BASE/$(folder_for "$s")/$s/$s.zip"
  if [ -d "$OUT/$s/mav0" ]; then echo "[skip] $s already extracted"; continue; fi
  echo "[get ] $url"
  curl -fL --retry 3 -o "$OUT/$s.zip" "$url"
  mkdir -p "$OUT/$s"
  unzip -q "$OUT/$s.zip" -d "$OUT/$s"
  rm -f "$OUT/$s.zip"
  echo "[done] $OUT/$s/mav0"
done
