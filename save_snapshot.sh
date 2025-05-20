#!/usr/bin/env bash
#
# save_snapshot.sh  –  Dump a folder’s tree plus every file’s contents
# Usage:
#   ./save_snapshot.sh [TARGET_DIR] [OUTPUT_TXT]
# Defaults:
#   TARGET_DIR="."                  (current directory)
#   OUTPUT_TXT="snapshot.txt"       (created/overwritten)

set -euo pipefail

###############################################################################
# 1. Input handling
###############################################################################
TARGET_DIR="${1:-.}"
OUTPUT_TXT="${2:-snapshot.txt}"

# Verify prerequisites
command -v tree >/dev/null 2>&1 || {
  printf 'Error: "tree" command not found. Please install it first.\n' >&2; exit 1; }

# Avoid accidental overwrite of important files
if [[ -e "$OUTPUT_TXT" && ! -w "$OUTPUT_TXT" ]]; then
  printf 'Error: Output file "%s" is not writable.\n' "$OUTPUT_TXT" >&2
  exit 1
fi

###############################################################################
# 2. Capture the directory tree
###############################################################################
# Truncate/overwrite existing output
: > "$OUTPUT_TXT"

printf '### Directory tree for: %s\n\n' "$TARGET_DIR" >> "$OUTPUT_TXT"
tree -F "$TARGET_DIR" >> "$OUTPUT_TXT" || {
  printf 'Warning: Unable to generate tree for "%s".\n' "$TARGET_DIR" >&2; }

printf '\n\n### File contents\n\n' >> "$OUTPUT_TXT"

###############################################################################
# 3. Append each file (relative path + contents)
###############################################################################
# Absolute path to output to compare properly
ABS_OUTPUT="$(realpath "$OUTPUT_TXT")"

# Use null-delimited pipeline, excluding the output file itself
find "$TARGET_DIR" -type f -print0 | sort -z |
while IFS= read -r -d '' FILE
do
  # Resolve real path for comparison
  ABS_FILE="$(realpath "$FILE")"

  # Skip the output file itself
  if [[ "$ABS_FILE" == "$ABS_OUTPUT" ]]; then
    continue
  fi

  # Remove the leading base path for readability
  REL_PATH="${FILE#$TARGET_DIR/}"

  # Header: path
  printf '%s\n' "$REL_PATH" >> "$OUTPUT_TXT"
  printf '"""\n' >> "$OUTPUT_TXT"

  # Body: raw file bytes (warn if unreadable)
  if ! cat "$FILE" >> "$OUTPUT_TXT" 2>/dev/null; then
      printf '[[[ Error reading file ]]]\n' >> "$OUTPUT_TXT"
      printf 'Warning: Could not read "%s".\n' "$REL_PATH" >&2
  fi

  # Footer
  printf '\n"""\n\n' >> "$OUTPUT_TXT"
done

printf 'Snapshot saved to "%s"\n' "$OUTPUT_TXT"
