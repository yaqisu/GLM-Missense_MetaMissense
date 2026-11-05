#!/bin/bash
# generate_datasets.sh
# Generates sequence TSV files from BED files and optionally splits them into
# train/val sets, as defined in a config file.
#
# For each row in the config:
#   - Extracts flanking sequences for each BED file with the appropriate label,
#     for the window sizes listed in the row's `sizes` column
#   - If split=yes: combines per-class TSVs and splits them into training / held-out
#     sets by the row's `heldout` chromosome set
#   - If split=no:  generates sequences only
#
# Usage:
#   bash preprocessing/generate_datasets.sh -c <config> [-s <sizes>]
#
# Options:
#   -c <config>     Path to config TSV (e.g. preprocessing/config.tsv). Required.
#                   Edit this file to add/change datasets. See README for format.
#   -s <sizes>      Optional. Comma-separated window sizes; only these sizes are
#                   generated (each row still only generates the sizes it lists).
#                   Default: every size listed in the config.
#   -h              Show this help message
#
# Examples:
#   bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv           # all Table 2 sets
#   bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 12k    # 12k only
#
# Window size math:
#   For a given size Xk: total_window = round(X * 1000), flank = (total_window - 1) / 2
#   The variant base occupies position 0; flank bp are extracted on each side.
#   Total sequence length = 2 * flank + 1.
#
#   Predefined sizes:
#     6k   -> flank 2999  -> 5999 bp total
#     12k  -> flank 5999  -> 11999 bp total
#     30k  -> flank 14999 -> 29999 bp total
#     60k  -> flank 29999 -> 59999 bp total
#     130k -> flank 64999 -> 129999 bp total

set -euo pipefail

GENOME="data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
SEQ_DIR="data/sequences"
SPLIT_DIR="data/splits"
SEQ_SCRIPT="preprocessing/extract_variant_sequences.py"
SPLIT_SCRIPT="preprocessing/split_data_fixed_chroms.py"
CONFIG=""

# Predefined sizes: maps suffix -> flank length
declare -A PRESET_LENGTHS=(
    ["6k"]=2999
    ["12k"]=5999
    ["30k"]=14999
    ["60k"]=29999
    ["130k"]=64999
)

# --- Parse arguments ---------------------------------------------------------
SIZES_ARG=""
while getopts ":c:s:h" opt; do
    case "${opt}" in
        c) CONFIG="${OPTARG}" ;;
        s) SIZES_ARG="${OPTARG}" ;;
        h) sed -n '/^# Usage/,/^# Window/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        :) echo "ERROR: Option -${OPTARG} requires an argument." >&2; exit 1 ;;
       \?) echo "ERROR: Unknown option -${OPTARG}." >&2; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: -c <config> is required. Use -h for help." >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$GENOME" ]] || { echo "ERROR: Reference genome not found: $GENOME" >&2; exit 1; }

# --- Resolve a size (predefined or arbitrary Xk) to its flank -----------------
# For a given Xk: total = round(X * 1000), flank = (total - 1) / 2
# We require total to be odd so the variant sits exactly in the middle.
# If round(X*1000) is even, we subtract 1 to make it odd.
flank_of() {
    local s="${1// /}"
    if [[ -v PRESET_LENGTHS[$s] ]]; then
        echo "${PRESET_LENGTHS[$s]}"
    elif [[ "$s" =~ ^([0-9]+\.?[0-9]*)k$ ]]; then
        python3 -c "
x = float('${BASH_REMATCH[1]}')
total = round(x * 1000)
if total % 2 == 0:
    total -= 1
print((total - 1) // 2)
"
    else
        echo "ERROR: Invalid size '$s'. Use a number followed by k, e.g. 12k or 11.7k." >&2
        exit 1
    fi
}

declare -A LENGTHS   # suffix -> flank, filled per config row
ONLY=",${SIZES_ARG// /},"   # sizes given with -s, comma-delimited ("" = no restriction)

mkdir -p "$SEQ_DIR" "$SPLIT_DIR"

# --- Helper: generate sequence TSV for one BED file --------------------------
generate_one_bed() {
    local label="$1"   # integer label, or "unlabeled"
    local suffix="$2"
    local bed="$3"

    [[ -z "$bed" ]] && return

    local base output
    base=$(basename "$bed")
    output="$SEQ_DIR/${base}.seq${suffix}.tsv"

    if [[ -f "$output" ]]; then
        echo "    Skipping (already exists): $output"
        return
    fi

    if [[ ! -f "$bed" ]]; then
        echo "    ERROR: BED file not found: $bed" >&2
        exit 1
    fi

    local flank="${LENGTHS[$suffix]}"
    local total=$(( flank * 2 + 1 ))

    if [[ "$label" == "unlabeled" ]]; then
        echo "    Generating (unlabeled): $output  [flank=${flank}, total=${total}bp]"
        python "$SEQ_SCRIPT" -b "$bed" -f "$GENOME" \
            -l "$flank" -o "$output"
    else
        echo "    Generating (label=$label): $output  [flank=${flank}, total=${total}bp]"
        python "$SEQ_SCRIPT" -b "$bed" -f "$GENOME" \
            -l "$flank" --label "$label" -o "$output"
    fi
}

# --- Helper: split combined TSV into train/val -------------------------------
split_dataset() {
    local name="$1"
    local suffix="$2"
    local -n _class0_beds="$3"
    local -n _class1_beds="$4"
    local heldout="$5"

    # Filename format: {prefix}.seq{suffix}.{mode}_{training|validation}.tsv
    # e.g. ClinVar.251103.missense.hg38.seq12k.BvsP_training.tsv
    local prefix="${name%.*}"
    local mode="${name##*.}"
    local train_out="$SPLIT_DIR/${prefix}.seq${suffix}.${mode}_training.tsv"
    local val_out="$SPLIT_DIR/${prefix}.seq${suffix}.${mode}_validation.tsv"

    if [[ -f "$train_out" && -f "$val_out" ]]; then
        echo "    Skipping split (already exists): ${prefix}.seq${suffix}.${mode}"
        return
    fi

    local combined header_written=0
    combined=$(mktemp)

    for bed in "${_class0_beds[@]:-}" "${_class1_beds[@]:-}"; do
        [[ -z "$bed" ]] && continue
        local tsv="$SEQ_DIR/$(basename "$bed").seq${suffix}.tsv"

        if [[ ! -f "$tsv" ]]; then
            echo "    ERROR: Sequence file not found: $tsv" >&2
            echo "    Run generate_datasets.sh -c $CONFIG -s $suffix first." >&2
            rm "$combined"; exit 1
        fi

        if [[ "$header_written" -eq 0 ]]; then
            cat "$tsv" >> "$combined"
            header_written=1
        else
            tail -n +2 "$tsv" >> "$combined"
        fi
    done

    echo "    Splitting: ${prefix}.seq${suffix}.${mode}  (held out: ${heldout})"
    python "$SPLIT_SCRIPT" --input "$combined" --train "$train_out" --val "$val_out" --heldout "$heldout"
    rm "$combined"
}

# --- Helper: concatenate all per-class TSVs into one file (for split=no datasets) ---
# Output: {SEQ_DIR}/{dataset_prefix}.seq{suffix}.tsv
# e.g. {dataset_prefix}.seq12k.tsv
# Includes all classes: class0, class1, and unlabeled if present.
concat_dataset() {
    local name="$1"
    local suffix="$2"
    local -n _c0_beds="$3"
    local -n _c1_beds="$4"
    local -n _ul_beds="$5"

    # Output name: strip the trailing .{mode} from name (not applicable for
    # split=no datasets, but handle gracefully), then use the raw name as prefix
    local prefix="${name%.*}"
    # If name has no dot-separated mode suffix, prefix == name — that's fine
    [[ "$prefix" == "$name" ]] && prefix="$name"
    local out="$SEQ_DIR/${prefix}.seq${suffix}.tsv"

    if [[ -f "$out" ]]; then
        echo "    Skipping concat (already exists): $out"
        return
    fi

    local header_written=0
    local n_files=0

    for bed in "${_c0_beds[@]:-}" "${_c1_beds[@]:-}" "${_ul_beds[@]:-}"; do
        [[ -z "$bed" ]] && continue
        local tsv="$SEQ_DIR/$(basename "$bed").seq${suffix}.tsv"

        if [[ ! -f "$tsv" ]]; then
            echo "    ERROR: Sequence file not found: $tsv" >&2
            exit 1
        fi

        if [[ "$header_written" -eq 0 ]]; then
            cat "$tsv" > "$out"
            header_written=1
        else
            tail -n +2 "$tsv" >> "$out"
        fi
        (( n_files++ )) || true
    done

    if [[ "$n_files" -eq 0 ]]; then
        echo "    WARNING: No input files found for concat, skipping." >&2
        return
    fi

    echo "    Concatenated ${n_files} class file(s) -> $out"
}

# --- Main loop over config rows ----------------------------------------------
echo "Config : $CONFIG"
[[ -n "$SIZES_ARG" ]] && echo "Sizes  : only ${SIZES_ARG}"
echo ""

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    # Use cut to parse fields — preserves empty fields unlike IFS read
    name=$(echo "$line"      | cut -f1)
    class0=$(echo "$line"    | cut -f2)
    class1=$(echo "$line"    | cut -f3)
    unlabeled=$(echo "$line" | cut -f4)
    split=$(echo "$line"     | cut -f5)
    heldout=$(echo "$line"   | cut -f6)
    sizes=$(echo "$line"     | cut -f7)

    [[ -z "$name" ]] && continue
    [[ "${split:-no}" == "yes" && -z "$heldout" ]] && {
        echo "ERROR: ${name}: split=yes needs a heldout chromosome set (column 6)." >&2; exit 1; }
    [[ -z "$sizes" ]] && { echo "ERROR: ${name}: no sizes (column 7)." >&2; exit 1; }

    echo "============================================================"
    echo "Dataset: ${name}  (split=${split:-no}, heldout=${heldout:--}, sizes=${sizes})"
    echo "============================================================"

    IFS=';' read -ra class0_beds    <<< "${class0:-}"
    IFS=';' read -ra class1_beds    <<< "${class1:-}"
    IFS=';' read -ra unlabeled_beds <<< "${unlabeled:-}"
    IFS=',' read -ra row_sizes      <<< "${sizes}"

    for suffix in "${row_sizes[@]}"; do
        suffix="${suffix// /}"
        if [[ -n "$SIZES_ARG" && "$ONLY" != *",${suffix},"* ]]; then continue; fi
        LENGTHS[$suffix]=$(flank_of "$suffix")
        echo "  -- seq${suffix} --"

        # Step 1: generate sequences
        for bed in "${class0_beds[@]:-}";    do generate_one_bed "0"         "$suffix" "$bed"; done
        for bed in "${class1_beds[@]:-}";    do generate_one_bed "1"         "$suffix" "$bed"; done
        for bed in "${unlabeled_beds[@]:-}"; do generate_one_bed "unlabeled" "$suffix" "$bed"; done

        # Step 2: split or concatenate
        if [[ "${split:-no}" == "yes" ]]; then
            split_dataset "$name" "$suffix" class0_beds class1_beds "$heldout"
        else
            concat_dataset "$name" "$suffix" class0_beds class1_beds unlabeled_beds
        fi
    done
    echo ""

done < "$CONFIG"

echo "Done!"
echo "  Sequences : $SEQ_DIR"
echo "  Splits    : $SPLIT_DIR"
