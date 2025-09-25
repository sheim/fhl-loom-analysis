#!/usr/bin/env bash
set -euo pipefail

# hard-coded target folders
folders=("circle" "ghost" "flapping" "fixed")

# directory to process (default = current dir)
target_dir="${1:-.}"

# ensure target_dir exists
if [[ ! -d "$target_dir" ]]; then
  echo "Error: $target_dir is not a directory"
  exit 1
fi

# create folders inside target_dir
for folder in "${folders[@]}"; do
  mkdir -p "$target_dir/$folder"
done

shopt -s nullglob nocaseglob
for f in "$target_dir"/Trial_*.MP4; do
  [[ -e "$f" ]] || continue  # skip if no matches

  # extract trial number
  base=$(basename "$f")
  if [[ "$base" =~ ^Trial_([0-9]+)_ ]]; then
    num="${BASH_REMATCH[1]}"
  else
    continue
  fi

  # case-insensitive folder match using tr
  lower_base=$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')

  for folder in "${folders[@]}"; do
    if [[ "$lower_base" == *"${folder}"* ]]; then
      dest="$target_dir/$folder/${num}.MP4"

      # add suffix if file already exists
      if [[ -e "$dest" ]]; then
        i=1
        while [[ -e "$target_dir/$folder/${num}_$i.MP4" ]]; do
          ((i++))
        done
        dest="$target_dir/$folder/${num}_$i.MP4"
      fi

      mv "$f" "$dest"
      echo "Moved: $base → $(basename "$dest")"
      break
    fi
  done
done
