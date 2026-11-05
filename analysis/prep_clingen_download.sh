#!/usr/bin/env bash
# Download the ClinGen Evidence Repository (ERepo): expert-panel variant classifications with the
# ACMG/AMP criteria met / not met. Run from the repo root:  bash analysis/prep_clingen_download.sh
set -euo pipefail

OUT=data/clingen
DATE=$(date +%Y-%m-%d)
FILE=$OUT/erepo.tabbed_${DATE}.txt
mkdir -p "$OUT"

URLS=(
  "https://erepo.clinicalgenome.org/evrepo/api/classifications/all?format=tabbed"
  "https://erepo.clinicalgenome.org/redmine/projects/evrepo/pcer/api/classifications/all?format=tabbed"
)

for u in "${URLS[@]}"; do
  echo "trying $u"
  if curl -fL --retry 3 -o "$FILE.tmp" "$u" && head -1 "$FILE.tmp" | grep -q 'Applied Evidence Codes'; then
    mv "$FILE.tmp" "$FILE"
    ln -sf "$(basename "$FILE")" "$OUT/erepo.tabbed.txt"   # stable name for downstream code
    echo "saved $FILE ($(($(wc -l < "$FILE") - 1)) classifications)"
    head -1 "$FILE" | tr '\t' '\n' | nl
    exit 0
  fi
  echo "  failed or not the expected table"
done
rm -f "$FILE.tmp"
echo "all URLs failed; download manually from https://erepo.clinicalgenome.org/evrepo/ (Download -> tab-delimited)" >&2
exit 1
