#!/usr/bin/env bash
# Copy the regenerated paper figures into the Overleaf clone and, with --push,
# commit and push them so they appear in the Overleaf project.
#
#   scripts/sync_figures_to_overleaf.sh          # copy only, show git status
#   scripts/sync_figures_to_overleaf.sh --push   # copy, commit, push
#
# The Overleaf project (https://www.overleaf.com/project/6a85dab8ba69b33284f57539)
# is cloned as a sibling of this repo: ../federated_grokking_paper. Regenerate
# the figures first with `python3 scripts/plotting/paper_figures.py`.
set -euo pipefail
cd "$(dirname "$0")/.."
OVERLEAF="${OVERLEAF:-../federated_grokking_paper}"
DEST="$OVERLEAF/figure/v2"
[ -d "$OVERLEAF/.git" ] || { echo "no Overleaf clone at $OVERLEAF"; exit 1; }
mkdir -p "$DEST"
cp paper/figures/fig*.pdf "$DEST/"
cp paper/figures/figure_numbers.json "$DEST/"
echo "copied $(ls "$DEST"/*.pdf | wc -l | tr -d ' ') PDFs to $DEST"
cd "$OVERLEAF"
git add figure/v2
git status --short figure/v2
if [ "${1:-}" = "--push" ]; then
  git commit -q -m "figures: regenerate from runs_v2 ($(date +%Y-%m-%d))" || echo "nothing to commit"
  git push
fi
