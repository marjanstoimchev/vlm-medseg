#!/usr/bin/env bash
# Offline-fallback delivery: push this repo as a Kaggle Dataset so the notebook
# can install it without internet (the primary path is the GitHub pip-install in
# the notebook's bootstrap cell). Requires the Kaggle CLI + ~/.kaggle/kaggle.json.
#
#   pip install kaggle
#   ./scripts/push_to_kaggle.sh            # create or version dataset
#
# In the notebook (offline): attach the dataset, then
#   !pip install -q /kaggle/input/vlm-medseg-src
set -euo pipefail

SLUG="${KAGGLE_DATASET:-marjan1111/vlm-medseg-src}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"

# Stage only the source the notebook needs (no data/runs/.git).
rsync -a --exclude='.git' --exclude='data' --exclude='runs' --exclude='outputs' \
      --exclude='__pycache__' --exclude='*.egg-info' \
      "$REPO_ROOT/src" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" "$STAGE/"

cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "vlm-medseg source",
  "id": "${SLUG}",
  "licenses": [{"name": "MIT"}]
}
JSON

if kaggle datasets status "$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update $(date -u +%FT%TZ)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
rm -rf "$STAGE"
echo "pushed $SLUG"
