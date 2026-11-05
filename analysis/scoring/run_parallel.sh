#!/usr/bin/env bash
# Saluki, sequence or FIMO features for every held-out VUS, in parallel on CPU. The VUS (test chromosomes) are split into
# N shards by transcript (a transcript's variants stay in one shard, so Saluki encodes each mRNA once), one process
# per shard. Saluki shards resume when rerun (variants already written are skipped); sequence and FIMO features are
# fast and simply rerun. Outputs: results/scores/vus_<kind>/<kind>_features_<i>.tsv.
#
#   N=50 THREADS=2 bash analysis/scoring/run_parallel.sh saluki   # 100 cores -> results/scores/vus_saluki/saluki_features_<i>.tsv
#   N=50 bash analysis/scoring/run_parallel.sh seq                 # -> results/scores/vus_seq/seq_features_<i>.tsv
#   N=50 bash analysis/scoring/run_parallel.sh fimo                # env meme -> results/scores/vus_fimo/fimo_features_<i>.tsv
# Keep N the same when rerunning to resume: the shards (transcript hash mod N) change with N. Each Saluki process loads
# every model (~1-2 GB of RAM each); N=100 THREADS=1 also uses 100 cores if the machine has the memory (free -g).
#
# Needs FASTA and GTF (Ensembl GRCh38 release 115); Saluki also SALUKI_MODELS (quoted glob) and SALUKI_PARAMS; FIMO
# only FASTA and MOTIFS (default: the JASPAR 2026 file of fimo_motifs.py).
# THREADS (default: cores / N) caps the threads of each Saluki process so the processes do not oversubscribe.
# Run from the repo root, conda env saluki (TensorFlow 2; saluki and seq) or meme (MEME suite; fimo). merge_vus.py
# then adds every shard to the annotated VUS table.
set -euo pipefail
KIND=${1:?usage: run_parallel.sh saluki|seq|fimo}
[[ $KIND == saluki || $KIND == seq || $KIND == fimo ]] || { echo "kind must be saluki, seq or fimo" >&2; exit 1; }
N=${N:-16}
THREADS=${THREADS:-$(( $(nproc) / N > 0 ? $(nproc) / N : 1 ))}
VUS=${VUS:-results/annotated/vus/ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv}
CSC=${CSC:-results/annotated/vus/reference/ORFeome_CSCs_wu2019.csv}          # codon stability (Wu et al. 2019)
ESRSEQ=${ESRSEQ:-results/annotated/vus/reference/ESRseq_scores_Ke2011.tsv}  # ESRseq hexamers (Ke et al. 2011)
MOTIFS=${MOTIFS:-data/external/jaspar2026/JASPAR2026_CORE_vertebrates_non-redundant.meme}   # fimo_motifs.py
: "${FASTA:?set FASTA}"
[[ $KIND == fimo ]] || : "${GTF:?set GTF}"
[[ $KIND != fimo || -f $MOTIFS ]] || { echo "no motif file $MOTIFS (MOTIFS; analysis/scoring/fimo_motifs.py)" >&2; exit 1; }
[[ $KIND != saluki ]] || : "${SALUKI_MODELS:?set SALUKI_MODELS}"
HERE=$(cd "$(dirname "$0")" && pwd)
D=results/scores/vus_$KIND
mkdir -p "$D/shards"

python - "$VUS" "$D/shards" "$N" <<'EOF'
import sys, zlib
import pandas as pd
vus, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = pd.read_csv(vus, sep='\t', dtype=str, keep_default_na=False,
                usecols=lambda c: c in {'variant_id', 'variant_id_x', 'TRANSCRIPT_ID', 'gene', 'genename'})
t = t.rename(columns={'variant_id_x': 'variant_id'}).drop_duplicates('variant_id')
t = t[t.variant_id.str.split(':').str[0].str.replace('^chr', '', regex=True).isin({'2', '3', '16', '17'})]
k = t.TRANSCRIPT_ID.map(lambda x: zlib.crc32(x.split('.')[0].encode()) % n)
for i in range(n):
    t[k == i].to_csv(f'{out}/shard_{i:03d}.tsv', sep='\t', index=False)
print(f'{len(t):,} held-out VUS in {n} shards ({out})')
EOF
python -c "import sys; sys.path.insert(0, '$HERE'); from seq_features import Fasta; Fasta('$FASTA')"   # .fai once

for i in $(seq -f %03g 0 $((N - 1))); do
  in=$D/shards/shard_$i.tsv
  if [[ $KIND == saluki ]]; then
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 \
      python "$HERE/saluki_score.py" --input "$in" --fasta "$FASTA" --gtf "$GTF" --models $SALUKI_MODELS \
      ${SALUKI_PARAMS:+--params "$SALUKI_PARAMS"} -o "$D/saluki_features_$i.tsv" > "$D/shards/log_$i.txt" 2>&1 &
  elif [[ $KIND == fimo ]]; then
    python "$HERE/fimo_score.py" --input "$in" --fasta "$FASTA" --motifs "$MOTIFS" \
      -o "$D/fimo_features_$i.tsv" > "$D/shards/log_$i.txt" 2>&1 &
  else
    EXTRA=()
    [[ -f $CSC ]] && EXTRA+=(--csc "$CSC")
    [[ -f $ESRSEQ ]] && EXTRA+=(--hexamer_table "$ESRSEQ")
    python "$HERE/seq_features.py" --input "$in" --fasta "$FASTA" --gtf "$GTF" ${EXTRA[@]+"${EXTRA[@]}"} \
      -o "$D/seq_features_$i.tsv" > "$D/shards/log_$i.txt" 2>&1 &
  fi
done
echo "$N $KIND processes started ($THREADS threads each); logs: $D/shards/log_*.txt"
wait
echo "done: $(ls "$D"/${KIND}_features_*.tsv | wc -l) shard outputs in $D"
grep -L "Wrote" "$D"/shards/log_*.txt 2>/dev/null | sed 's/^/  check (no "Wrote" line): /' || true
