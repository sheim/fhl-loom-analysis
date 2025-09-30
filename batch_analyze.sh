#!/usr/bin/env bash
# batch_analyze.sh
# Usage: ./batch_analyze.sh videos/circle

INPUT_DIR="$1"

# OUT_CSV="batch_results.csv"

# OUT_CSV file named with input_dir name
OUT_CSV="$(basename "$INPUT_DIR")_results.csv"

if [ -z "$INPUT_DIR" ]; then
  echo "Usage: $0 <video-folder>"
  exit 1
fi

echo "filename,stim_idx,final_det_idx" > "$OUT_CSV"

for vid in "$INPUT_DIR"/*.MP4; do
  echo "Processing $vid..."
  # Run the analyzer and capture its stdout
  OUTPUT=$(uv run analyze_fish_energy.py "$vid")

  # Extract stim_idx and final_det_idx from the printed lines
  stim=$(echo "$OUTPUT" | grep "Stimulus frame index" | awk '{print $4}')
  coarse=$(echo "$OUTPUT" | grep "Coarse first-movement" | awk '{print $4}')
  refined=$(echo "$OUTPUT" | grep "Refined first-movement" | awk '{print $4}')

  # Prefer refined if available, otherwise coarse
  if [ -n "$refined" ]; then
    det="$refined"
  else
    det="$coarse"
    echo "(Note: Using coarse detection for $vid)"
  fi

  # Append to CSV
  fname=$(basename "$vid")
  echo "$fname,$stim,$det" >> "$OUT_CSV"
done

echo "All done. Results saved to $OUT_CSV"
