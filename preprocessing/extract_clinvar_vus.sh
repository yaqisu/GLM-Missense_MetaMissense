#!/usr/bin/env bash
# =============================================================================
# extract_clinvar_vus.sh
# Extract missense VUS (and optionally conflicting) variants from ClinVar
# releases into BED, restricted by default to the chromosomes held out from
# GLM-Missense training. Optionally annotates each VUS with its classification
# in a later release (to find VUS that have since been reclassified).
#
# Usage:
#   ./extract_clinvar_vus.sh [-t <ts1,ts2,...>] [-c <chroms|all>] [-l <later_ts>]
#                            [-C] [-m] [-b <bcftools_path>]
#
# Options:
#   -t <timestamps>    Comma-separated ClinVar timestamps (default: clinvar_20251103).
#   -c <chroms|all>    Chromosomes to keep (default: 2,3,16,17 = GLM-Missense
#                      held-out set). Use "all" for genome-wide.
#   -l <timestamp>     Later release used to annotate reclassification of each VUS
#                      (e.g. clinvar_20260923). Writes an extra TSV.
#   -C                 Also extract "Conflicting_*" classifications.
#   -m                 Also keep VUS carrying secondary terms, e.g.
#                      "Uncertain_significance|risk_factor" (default: exact match only,
#                      consistent with the P/LP/B/LB sets; the excluded count is logged).
#   -b <path>          Path to bcftools binary (default: bcftools on PATH).
#   -h                 Show this help message.
#
# Example:
#   ./extract_clinvar_vus.sh -t clinvar_20251103 -l clinvar_20260923 -C
#
# Older releases are fetched from vcf_GRCh38/archive_2.0/<year>/ (monthly releases
# only); if a stem is not found, the stems available for that year are listed.
#
# Reclassification groups in the TSV: P_LP / B_LB / VUS = exact ClinVar terms;
# *_mod = same term with a secondary term or modifier (e.g. "|risk_factor",
# ",_low_penetrance"); conflicting; other; absent (not in the later release).
#
# Outputs (BED: same 6-column format as process_clinvar.sh):
#   data/bed/ClinVar.<tag>.missense.hg38.uncertain_significance[.chr2_3_16_17].bed
#   data/bed/ClinVar.<tag>.missense.hg38.conflicting[.chr2_3_16_17].bed        (-C)
#   data/bed/ClinVar.<tag>.missense.hg38.uncertain_significance[...].reclass_<later_tag>.tsv (-l)
# =============================================================================

set -o pipefail

BASE_URLS=("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly"
           "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38")
ARCHIVE_URL="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0"
BCFTOOLS="bcftools"
VCF_DIR="data/vcf"
BED_DIR="data/bed"
TIMESTAMPS=()
CHROMS="2,3,16,17"
LATER=""
WITH_CONFLICTING=0
WITH_MODIFIED=0

VUS_EXPR='INFO/CLNSIG="Uncertain_significance"'
VUS_MOD_EXPR='INFO/CLNSIG~"^Uncertain_significance[|,]"'     # secondary term / modifier
# "interpretations" in older releases, "classifications" in newer ones
CONFLICT_EXPR='INFO/CLNSIG~"^Conflicting_"'

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '/^# Usage/,/^# ====/p' "$0" | sed 's/^# \?//'
    exit 0
}

parse_args() {
    while getopts ":t:c:l:b:Cmh" opt; do
        case "${opt}" in
            t) IFS=',' read -ra _ts <<< "${OPTARG}"
               TIMESTAMPS+=("${_ts[@]}") ;;
            c) CHROMS="${OPTARG}" ;;
            l) LATER="${OPTARG}" ;;
            b) BCFTOOLS="${OPTARG}" ;;
            C) WITH_CONFLICTING=1 ;;
            m) WITH_MODIFIED=1 ;;
            h) usage ;;
            :) die "Option -${OPTARG} requires an argument." ;;
            \?) die "Unknown option: -${OPTARG}. Use -h for help." ;;
        esac
    done
    [[ ${#TIMESTAMPS[@]} -eq 0 ]] && TIMESTAMPS=("clinvar_20251103")
    if [[ ${WITH_MODIFIED} -eq 1 ]]; then
        VUS_EXPR='INFO/CLNSIG="Uncertain_significance" || INFO/CLNSIG~"^Uncertain_significance[|,]"'
        VUS_LABEL="uncertain_significance_incl_mod"
    else
        VUS_LABEL="uncertain_significance"
    fi

    # ClinVar VCFs use 1..22, X, Y, MT (no "chr" prefix)
    if [[ "${CHROMS}" == "all" ]]; then
        TARGET_ARGS=()
        SUFFIX=""
    else
        CHROMS=$(echo "${CHROMS}" | sed 's/chr//g; s/ //g')
        TARGET_ARGS=(-t "${CHROMS}")
        SUFFIX=".chr$(echo "${CHROMS}" | tr ',' '_')"
    fi
}

check_deps() {
    for cmd in wget bgzip md5sum; do
        command -v "${cmd}" &>/dev/null || die "'${cmd}' not found. Please install it."
    done
    "${BCFTOOLS}" --version &>/dev/null \
        || die "bcftools not runnable at '${BCFTOOLS}'. Use -b with the path to the binary."
}

# -----------------------------------------------------------------------------
# Download (identical logic/paths to process_clinvar.sh, so existing files are reused)
# -----------------------------------------------------------------------------
download_vcf() {
    local stem="$1"
    local gz="${VCF_DIR}/${stem}.vcf.gz"

    log "--- Downloading ${stem} ---"
    if [[ -f "${gz}" && -f "${gz}.tbi" ]]; then
        log "  Already exists, skipping: ${gz}"
    else
        local base year
        year=$(echo "${stem}" | sed -E 's/^clinvar_([0-9]{4}).*/\1/')
        for base in "${BASE_URLS[@]}" "${ARCHIVE_URL}/${year}"; do
            if wget -q --spider "${base}/${stem}.vcf.gz"; then
                log "  Fetching ${stem}.vcf.gz{,.tbi,.md5} from ${base}"
                for ext in vcf.gz vcf.gz.tbi vcf.gz.md5; do
                    wget -q --show-progress -O "${VCF_DIR}/${stem}.${ext}" "${base}/${stem}.${ext}"
                done
                break
            fi
        done
        if [[ ! -f "${gz}" ]]; then
            local avail
            avail=$(wget -qO- "${ARCHIVE_URL}/${year}/" \
                    | grep -oE 'clinvar_[0-9]{8}\.vcf\.gz' | sed 's/\.vcf\.gz//' | sort -u | tr '\n' ' ')
            die "${stem}.vcf.gz not found (weekly/, current, archive_2.0/${year}/). Releases archived for ${year}: ${avail:-none listed}"
        fi
    fi

    if [[ -f "${gz}.md5" ]]; then
        local expected actual
        expected=$(grep -oE '[0-9a-f]{32}' "${gz}.md5" | head -1)
        actual=$(md5sum "${gz}" | cut -d' ' -f1)
        [[ "${expected}" == "${actual}" ]] || die "md5 mismatch for ${gz}"
        log "  md5 OK"
    fi
}

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
}

# -----------------------------------------------------------------------------
# Extract one classification into BED (SNVs only, same format as process_clinvar.sh)
#   $1 = stem, $2 = tag, $3 = bcftools -i expression, $4 = label
# -----------------------------------------------------------------------------
extract_bed() {
    local stem="$1" tag="$2" expr="$3" label="$4"
    local src="${VCF_DIR}/${stem}_missense.vcf.gz"
    local dest="${BED_DIR}/ClinVar.${tag}.missense.hg38.${label}${SUFFIX}.bed"

    if [[ -f "${dest}" ]]; then
        log "    BED already exists, skipping: $(basename "${dest}")"
        return
    fi
    log "    Extracting ${label} -> $(basename "${dest}")"
    "${BCFTOOLS}" view "${TARGET_ARGS[@]}" -i "${expr}" "${src}" \
        | awk 'BEGIN{OFS="\t"} !/^#/ {
            if (length($4)==1 && length($5)==1)
                print "chr"$1, $2-1, $2, $3, $4, $5
          }' \
        | sort -k1,1V -k2,2n \
        > "${dest}.tmp" \
        && mv "${dest}.tmp" "${dest}" \
        || { rm -f "${dest}.tmp"; die "BED extraction failed for ${label} (${src})"; }
    log "    $(wc -l < "${dest}") variants written to $(basename "${dest}")"
}

# -----------------------------------------------------------------------------
# Annotate each VUS SNV with its classification in a later release
#   $1 = base stem, $2 = base tag, $3 = later stem, $4 = later tag
# -----------------------------------------------------------------------------
annotate_reclassification() {
    local stem="$1" tag="$2" later="$3" ltag="$4"
    local base_src="${VCF_DIR}/${stem}_missense.vcf.gz"
    local later_src="${VCF_DIR}/${later}.vcf.gz"   # full VCF: MC annotation may have changed
    local dest="${BED_DIR}/ClinVar.${tag}.missense.hg38.${VUS_LABEL}${SUFFIX}.reclass_${ltag}.tsv"
    local tmp_later="${dest}.later.tmp"

    if [[ -f "${dest}" ]]; then
        log "    Reclassification table already exists, skipping: $(basename "${dest}")"
        return
    fi
    log "    Annotating VUS with ${later} classification -> $(basename "${dest}")"

    "${BCFTOOLS}" view "${TARGET_ARGS[@]}" "${later_src}" \
        | "${BCFTOOLS}" query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\n' \
        > "${tmp_later}" \
        || { rm -f "${tmp_later}"; die "query failed for ${later_src}"; }

    "${BCFTOOLS}" view "${TARGET_ARGS[@]}" -i "${VUS_EXPR}" "${base_src}" \
        | "${BCFTOOLS}" query -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%INFO/GENEINFO\t%INFO/CLNREVSTAT\n' \
        | awk -F'\t' -v OFS='\t' -v btag="${tag}" -v ltag="${ltag}" '
            function group(s,   t, g, mod) {
                if (s == "." || s == "") return "absent"
                if (s ~ /^Conflicting_/) return "conflicting"
                t = s; sub(/\|.*/, "", t); sub(/,_.*/, "", t)
                mod = (t != s) ? "_mod" : ""
                if (t ~ /^(Pathogenic|Likely_pathogenic|Pathogenic\/Likely_pathogenic)$/) g = "P_LP"
                else if (t ~ /^(Benign|Likely_benign|Benign\/Likely_benign)$/) g = "B_LB"
                else if (t == "Uncertain_significance") g = "VUS"
                else return "other"
                return g mod
            }
            NR == FNR { later[$1 ":" $2 ":" $3 ":" $4] = $5 "\t" $6; next }
            FNR == 1 {
                print "chrom", "pos", "clinvar_id", "ref", "alt", "gene",
                      "revstat_" btag, "clnsig_" ltag, "revstat_" ltag, "group_" ltag
            }
            length($4) == 1 && length($5) == 1 {
                k = $1 ":" $2 ":" $4 ":" $5
                if (k in later) split(later[k], a, "\t"); else { a[1] = "."; a[2] = "." }
                sub(/:.*/, "", $6)   # GENEINFO "BRCA1:672|..." -> "BRCA1"
                print "chr" $1, $2, $3, $4, $5, $6, $7, a[1], a[2], group(a[1])
            }' "${tmp_later}" - \
        > "${dest}.tmp" \
        && mv "${dest}.tmp" "${dest}" \
        || { rm -f "${dest}.tmp" "${tmp_later}"; die "reclassification annotation failed"; }
    rm -f "${tmp_later}"

    log "    Status in ${later} of the $(($(wc -l < "${dest}") - 1)) VUS:"
    tail -n +2 "${dest}" | cut -f10 | sort | uniq -c | sort -k1,1nr \
        | while read -r n g; do log "      ${g}: ${n}"; done
}

date_tag() { echo "$1" | sed 's/clinvar_20//'; }

# =============================================================================
# Main
# =============================================================================
main() {
    parse_args "$@"
    check_deps

    log "bcftools   : ${BCFTOOLS}"
    log "Timestamps : ${TIMESTAMPS[*]}"
    log "Chromosomes: ${CHROMS}"
    [[ -n "${LATER}" ]] && log "Later rel. : ${LATER}"
    echo

    mkdir -p "${VCF_DIR}" "${BED_DIR}"

    if [[ -n "${LATER}" ]]; then
        download_vcf "${LATER}"
    fi

    for stem in "${TIMESTAMPS[@]}"; do
        local tag
        tag=$(date_tag "${stem}")
        log "====== Processing ${stem} (tag: ${tag}) ======"

        download_vcf "${stem}"
        extract_missense "${stem}"
        extract_bed "${stem}" "${tag}" "${VUS_EXPR}" "${VUS_LABEL}"
        local n_mod
        n_mod=$("${BCFTOOLS}" view -H "${TARGET_ARGS[@]}" -i "${VUS_MOD_EXPR}" \
                "${VCF_DIR}/${stem}_missense.vcf.gz" | awk 'length($4)==1 && length($5)==1' | wc -l)
        if [[ ${WITH_MODIFIED} -eq 1 ]]; then
            log "    (includes ${n_mod} VUS SNVs with secondary terms, e.g. |risk_factor)"
        else
            log "    (excluded ${n_mod} VUS SNVs with secondary terms, e.g. |risk_factor; use -m to keep)"
        fi
        [[ ${WITH_CONFLICTING} -eq 1 ]] \
            && extract_bed "${stem}" "${tag}" "${CONFLICT_EXPR}" "conflicting"

        if [[ -n "${LATER}" && "${LATER}" != "${stem}" ]]; then
            annotate_reclassification "${stem}" "${tag}" "${LATER}" "$(date_tag "${LATER}")"
        fi

        log "====== Finished ${stem} ======"
        echo
    done
    log "All done. BED files: ${BED_DIR}/"
}

main "$@"
