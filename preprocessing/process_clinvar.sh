#!/usr/bin/env bash
# =============================================================================
# process_clinvar.sh
# Download and process ClinVar VCF files for given timestamps, extracting
# missense variants by clinical significance into BED format.
#
# Usage:
#   ./process_clinvar.sh -t <ts1,ts2,...> [-b <bcftools_path>]
#
# Options:
#   -t <timestamps>    Comma-separated ClinVar timestamps to process.
#                      e.g. clinvar_20251103,clinvar_20260923
#   -b <path>          Path to bcftools binary (default: bcftools on PATH).
#   -h                 Show this help message.
#
# Example:
#   ./process_clinvar.sh \
#       -t clinvar_20251103,clinvar_20260923 \
#       -b /h/jenniferlin/Programs/bcftools/bin/bcftools
# =============================================================================

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
set -o pipefail   # a failing bcftools must not leave an empty output file

BASE_URLS=("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly"
           "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38")
BCFTOOLS="bcftools"
VCF_DIR="data/vcf"
BED_DIR="data/bed"
TIMESTAMPS=()

CLNSIG_LABELS=(
    "Pathogenic:pathogenic"
    "Likely_pathogenic:likely_pathogenic"
    "Benign:benign"
    "Likely_benign:likely_benign"
    "Benign/Likely_benign:benign_likely_benign"
    "Pathogenic/Likely_pathogenic:pathogenic_likely_pathogenic"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '/^# Usage/,/^# ====/p' "$0" | sed 's/^# \?//'
    exit 0
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
parse_args() {
    while getopts ":t:b:h" opt; do
        case "${opt}" in
            t) IFS=',' read -ra _ts <<< "${OPTARG}"
               TIMESTAMPS+=("${_ts[@]}") ;;
            b) BCFTOOLS="${OPTARG}" ;;
            h) usage ;;
            :) die "Option -${OPTARG} requires an argument." ;;
            \?) die "Unknown option: -${OPTARG}. Use -h for help." ;;
        esac
    done

    [[ ${#TIMESTAMPS[@]} -eq 0 ]] && die "-t <timestamps> is required. Use -h for help."
}

check_deps() {
    for cmd in wget bgzip; do
        command -v "${cmd}" &>/dev/null || die "'${cmd}' not found. Please install it."
    done
    # bcftools may be an absolute path or a name on PATH; must actually run
    "${BCFTOOLS}" --version &>/dev/null \
        || die "bcftools not runnable at '${BCFTOOLS}'. Use -b with the path to the binary (e.g. .../bin/bcftools)."
}

# -----------------------------------------------------------------------------
# Download a single timestamp
#   $1 = timestamp stem, e.g. clinvar_20251103
# -----------------------------------------------------------------------------
download_vcf() {
    local stem="$1"
    local gz="${VCF_DIR}/${stem}.vcf.gz"

    log "--- Downloading ${stem} ---"

    if [[ -f "${gz}" && -f "${gz}.tbi" ]]; then
        log "  Already exists, skipping: ${gz}"
    else
        # Try each FTP location (the current release sits in vcf_GRCh38/, older
        # weekly releases in weekly/); take all files from the first that has it.
        local base
        for base in "${BASE_URLS[@]}"; do
            if wget -q --spider "${base}/${stem}.vcf.gz"; then
                log "  Fetching ${stem}.vcf.gz{,.tbi,.md5} from ${base}"
                for ext in vcf.gz vcf.gz.tbi vcf.gz.md5; do
                    wget -q --show-progress -O "${VCF_DIR}/${stem}.${ext}" "${base}/${stem}.${ext}"
                done
                break
            fi
        done
        [[ -f "${gz}" ]] || die "${stem}.vcf.gz not found at: ${BASE_URLS[*]}"
    fi

    # Verify checksum when the .md5 is available
    if [[ -f "${gz}.md5" ]]; then
        local expected actual
        expected=$(grep -oE '[0-9a-f]{32}' "${gz}.md5" | head -1)
        actual=$(md5sum "${gz}" | cut -d' ' -f1)
        [[ "${expected}" == "${actual}" ]] || die "md5 mismatch for ${gz}"
        log "  md5 OK"
    fi
}

# -----------------------------------------------------------------------------
# Extract missense variants
#   $1 = timestamp stem
# -----------------------------------------------------------------------------
extract_missense() {
    local stem="$1"
    local src="${VCF_DIR}/${stem}.vcf.gz"
    local dest="${VCF_DIR}/${stem}_missense.vcf.gz"

    if [[ -f "${dest}" ]]; then
        log "  Missense VCF already exists, skipping: ${dest}"
        return
    fi

    log "  Extracting missense variants -> $(basename "${dest}")"
    "${BCFTOOLS}" view -i 'INFO/MC ~ "missense"' "${src}" | bgzip > "${dest}.tmp" \
        && mv "${dest}.tmp" "${dest}" \
        || { rm -f "${dest}.tmp"; die "missense extraction failed for ${src}"; }
    log "  Done: $(basename "${dest}")"
}

# -----------------------------------------------------------------------------
# Extract one clinical-significance category into BED
#   $1 = timestamp stem (e.g. clinvar_20251103)
#   $2 = date tag for filenames (e.g. 251103)
#   $3 = CLNSIG value (e.g. Pathogenic)
#   $4 = label for filename (e.g. pathogenic)
# -----------------------------------------------------------------------------
extract_bed() {
    local stem="$1"
    local tag="$2"
    local clnsig="$3"
    local label="$4"

    local src="${VCF_DIR}/${stem}_missense.vcf.gz"
    local dest="${BED_DIR}/ClinVar.${tag}.missense.hg38.${label}.bed"

    if [[ -f "${dest}" ]]; then
        log "    BED already exists, skipping: $(basename "${dest}")"
        return
    fi

    log "    Extracting ${clnsig} -> $(basename "${dest}")"
    "${BCFTOOLS}" view -i "INFO/CLNSIG=\"${clnsig}\"" "${src}" \
        | awk 'BEGIN{OFS="\t"} !/^#/ {
            if (length($4)==1 && length($5)==1)
                print "chr"$1, $2-1, $2, $3, $4, $5
          }' \
        | sort -k1,1V -k2,2n \
        > "${dest}.tmp" \
        && mv "${dest}.tmp" "${dest}" \
        || { rm -f "${dest}.tmp"; die "BED extraction failed for ${clnsig} (${src})"; }

    local n
    n=$(wc -l < "${dest}")
    log "    ${n} variants written to $(basename "${dest}")"
}

# -----------------------------------------------------------------------------
# Derive a short date tag from the stem (clinvar_20251103 -> 251103)
# -----------------------------------------------------------------------------
date_tag() {
    local stem="$1"
    echo "${stem}" | sed 's/clinvar_20//'
}

# =============================================================================
# Main
# =============================================================================
main() {
    parse_args "$@"
    check_deps

    log "bcftools  : ${BCFTOOLS}"
    log "Timestamps: ${TIMESTAMPS[*]}"
    log "VCF dir   : ${VCF_DIR}"
    log "BED dir   : ${BED_DIR}"
    echo

    mkdir -p "${VCF_DIR}" "${BED_DIR}"

    for stem in "${TIMESTAMPS[@]}"; do
        local tag
        tag=$(date_tag "${stem}")

        log "====== Processing ${stem} (tag: ${tag}) ======"

        # 1. Download
        download_vcf "${stem}"

        # 2. Extract missense
        extract_missense "${stem}"

        # 3. Extract each CLNSIG category
        for entry in "${CLNSIG_LABELS[@]}"; do
            local clnsig="${entry%%:*}"
            local label="${entry##*:}"
            extract_bed "${stem}" "${tag}" "${clnsig}" "${label}"
        done

        log "====== Finished ${stem} ======"
        echo
    done

    log "All done."
    log "VCF files : ${VCF_DIR}/"
    log "BED files : ${BED_DIR}/"
}

main "$@"