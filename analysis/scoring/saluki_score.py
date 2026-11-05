#!/usr/bin/env python3
"""
Saluki (Agarwal & Kelley 2022, Genome Biol) predicted change in mRNA half-life for SNVs on their
(MANE) transcript: delta = Saluki(alt mRNA) - Saluki(ref mRNA), averaged over the models given.

Encoding (as in basenji/bin/saluki_ssd.py, the authors' variant scorer, and the training TFRecords):
  mature mRNA (exons in transcript order, sense strand), 12,288 x 6:
    channels 0-3  one-hot A, C, G, T
    channel 4     1 at the first base of every codon of the CDS (GTF CDS features; only when the CDS
                  length is a multiple of 3)
    channel 5     splice track, --splice_encoding:
                    saluki_ssd (default)  as the authors' code writes it: exon lengths taken as end - start
                                          on 1-based GTF coordinates, exons in genomic order, so the marks
                                          drift by one base per exon, and the transcript's last base is
                                          marked only for minus-strand genes. The training TFRecords show
                                          this pattern (last base marked in ~half of the records); confirm
                                          with --inspect_tfr ... --fasta --gtf, which matches training
                                          records to transcripts and reports which encoding reproduces them.
                    exact                 last base of every exon
                    exact_internal        last base of every exon except the final one
  left-justified, zero-padded at the 3' end; transcripts longer than 12,288 nt keep their last 12,288
  nt (5' trimming), so a variant in the trimmed 5' part is not scored (status outside_window).
  Use --inspect_tfr on a training TFRecord to check these conventions on the authors' data (add --fasta
  and --gtf to compare the splice and frame tracks transcript by transcript).

Models: model0_best.h5 = head 0 (human) from datasets/deeplearning/train_gru/f{0-9}_c{0-4}/train/
(Zenodo 10.5281/zenodo.6326409; 10 folds x 5 runs = 50 models) or the single f0_c0 model of the
HuggingFace OneScience-Group/Saluki repackaging. The network is rebuilt here (same layers as
basenji rnann.RnaNN), so basenji itself is not needed; TensorFlow 2.x is. If loading fails under
Keras 3, use TensorFlow <= 2.15, or install tf_keras and set TF_USE_LEGACY_KERAS=1.

Inputs:
  --input    TSV with variant_id (chrN:pos:REF>ALT, GRCh38) and TRANSCRIPT_ID (version ignored)
  --fasta    GRCh38 FASTA, uncompressed (.fai built if missing)
  --gtf      GTF with exon and CDS features (MANE / Ensembl / GENCODE)
  --models   one or more .h5 files or quoted globs
  --params   params.json of the models (default: the manuscript's train_gru settings)
Needs seq_features.py in the same directory (FASTA and GTF readers).

Output (one row per variant; rerunning with the same -o resumes, skipping variants already written):
  saluki_status, saluki_tx_length, saluki_tx_pos (1-based in the mRNA), saluki_trimmed,
  saluki_ref, saluki_alt (model-mean predictions), saluki_delta (mean of alt - ref),
  saluki_delta_sd (across models), saluki_n_models [, saluki_delta_m<i> with --per_model]
Units are those of the training target (first principal component of standardised log half-lives);
negative delta = less stable mRNA.

Example:
    python saluki_score.py --input alphaGenome_annotated/selected_variants.tsv \
        --fasta GRCh38.fa --gtf Homo_sapiens.GRCh38.113.gtf.gz \
        --models 'train_gru/f*_c*/train/model0_best.h5' --params train_gru/params.json \
        -o alphaGenome_annotated/saluki_features.tsv
    python saluki_score.py --inspect_tfr train_gru/f0_c0/data0/tfrecords/test-0.tfr \
        [--fasta GRCh38.fa --gtf Homo_sapiens.GRCh38.113.gtf.gz]
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from seq_features import Fasta, read_gtf, norm_chrom, revcomp
except ImportError:
    sys.exit("seq_features.py must be in the same directory as this script")

SEQ_LEN = 12288
NT = {"A": 0, "C": 1, "G": 2, "T": 3}
DEFAULT_PARAMS = {"activation": "relu", "rnn_type": "gru", "seq_length": 12288, "num_targets": 1,
                  "filters": 64, "kernel_size": 5, "dropout": 0.3, "ln_epsilon": 0.007,
                  "num_layers": 6, "bn_momentum": 0.90}
STOPS = {"TAA", "TAG", "TGA"}


# ----------------------------------------------------------------------------- model
def build_saluki(p):
    """Same layer sequence as basenji rnann.RnaNN (residual=False, go_backwards=True), one head."""
    import tensorflow as tf
    L = tf.keras.layers
    if p.get("residual", False) or p.get("rnn_type", "gru") != "gru":
        sys.exit("only residual=False, rnn_type=gru (the published Saluki models) are supported")
    act = {"relu": L.ReLU, "gelu": lambda: L.Activation("gelu")}.get(p.get("activation", "relu"))
    if act is None:
        sys.exit(f"activation {p['activation']} not supported")
    f, k, d, eps = p["filters"], p["kernel_size"], p["dropout"], p["ln_epsilon"]
    nt = p.get("num_targets", 1)
    nt = nt[0] if isinstance(nt, list) else nt
    inp = tf.keras.Input(shape=(p["seq_length"], p.get("seq_depth", 6)), name="sequence")
    x = L.Conv1D(f, k, padding="valid", use_bias=False)(inp)
    for _ in range(p["num_layers"]):
        x = L.LayerNormalization(epsilon=eps)(x)
        x = act()(x)
        x = L.Conv1D(f, k, padding="valid")(x)
        x = L.Dropout(d)(x)
        x = L.MaxPooling1D()(x)
    x = L.LayerNormalization(epsilon=eps)(x)
    x = act()(x)
    x = L.GRU(f, go_backwards=p.get("go_backwards", True))(x)
    x = L.BatchNormalization(momentum=p["bn_momentum"])(x)
    x = act()(x)
    x = L.Dense(f)(x)
    x = L.Dropout(d)(x)
    x = L.BatchNormalization(momentum=p["bn_momentum"])(x)
    x = act()(x)
    out = L.Dense(nt)(x)
    return tf.keras.Model(inp, out)


def load_models(paths, params):
    models = []
    for pth in paths:
        m = build_saluki(params)
        try:
            m.load_weights(pth)
        except Exception as e:                                       # noqa: BLE001
            sys.exit(f"Could not load {pth}: {e}\nTry TensorFlow <= 2.15, or tf_keras with "
                     f"TF_USE_LEGACY_KERAS=1.")
        models.append(m)
    return models


# ----------------------------------------------------------------------------- encoding
def tx_index(t, pos):
    """0-based mRNA index of a genomic position, or None if not exonic."""
    cum = 0
    for s, e in t["exons"]:
        if s <= pos <= e:
            return cum + (pos - s if t["strand"] == "+" else e - pos)
        cum += e - s + 1
    return None


SPLICE_MODES = ("saluki_ssd", "exact", "exact_internal")


def splice_track(t, n, mode):
    """Splice channel (length n, transcript orientation)."""
    ch = np.zeros(n, dtype=np.float32)
    if mode == "saluki_ssd":                    # replicates rna_1hot_splice() in basenji saluki_ssd.py
        ti = 0
        for s, e in sorted(t["exons"]):         # genomic order, length end - start
            el = e - s
            if t["strand"] == "+":
                ti += el
                if 0 <= ti - 1 < n:
                    ch[ti - 1] = 1
            else:
                if 0 <= ti < n:
                    ch[ti] = 1
                ti += el
        return ch if t["strand"] == "+" else ch[::-1].copy()
    cum, k_last = 0, len(t["exons"]) - 1
    for k, (s, e) in enumerate(t["exons"]):     # transcript order
        cum += e - s + 1
        if k < k_last or mode == "exact":
            ch[cum - 1] = 1
    return ch


def frame_track(t, mrna, include_stop=False):
    ch = np.zeros(len(mrna), dtype=np.float32)
    if t["cds"]:
        plus = t["strand"] == "+"
        c0 = tx_index(t, t["cds"][0][0] if plus else t["cds"][0][1])
        clen = sum(e - s + 1 for s, e in t["cds"])
        if c0 is not None and include_stop and mrna[c0 + clen: c0 + clen + 3] in STOPS:
            clen += 3
        if c0 is not None and clen % 3 == 0:
            ch[c0:c0 + clen:3] = 1
    return ch


def mrna_seq(fa, t):
    parts = [fa.fetch(t["chrom"], s, e) for s, e in t["exons"]]
    return "".join(parts if t["strand"] == "+" else [revcomp(p) for p in parts])


def encode(fa, t, splice_mode="saluki_ssd", include_stop=False):
    """mRNA string and its L x 6 encoding (float32, not yet trimmed/padded)."""
    mrna = mrna_seq(fa, t)
    n = len(mrna)
    x = np.zeros((n, 6), dtype=np.float32)
    idx = np.array([NT.get(b, -1) for b in mrna])
    ok = idx >= 0
    x[np.where(ok)[0], idx[ok]] = 1
    x[~ok, :4] = 0.25
    x[:, 4] = frame_track(t, mrna, include_stop)
    x[:, 5] = splice_track(t, n, splice_mode)
    return mrna, x


def fit_window(x):
    """Trim 5' / pad 3' to SEQ_LEN; returns array and offset removed from the 5' end."""
    n = len(x)
    if n >= SEQ_LEN:
        return x[n - SEQ_LEN:], n - SEQ_LEN
    return np.concatenate([x, np.zeros((SEQ_LEN - n, x.shape[1]), np.float32)]), 0


# ----------------------------------------------------------------------------- TFRecord check
def inspect_tfr(paths, n_max, fasta=None, gtf=None):
    import tensorflow as tf
    spec = {"lengths": tf.io.FixedLenFeature((1,), tf.int64),
            "sequence": tf.io.FixedLenFeature([], tf.string),
            "coding": tf.io.FixedLenFeature([], tf.string),
            "splice": tf.io.FixedLenFeature([], tf.string)}
    rows, recs = [], []
    for pth in paths:
        for rec in tf.data.TFRecordDataset(pth, compression_type="ZLIB"):
            ex = tf.io.parse_single_example(rec, spec)
            L = int(ex["lengths"][0])
            seq = np.frombuffer(ex["sequence"].numpy(), np.uint8)
            cod = np.frombuffer(ex["coding"].numpy(), np.uint8)
            spl = np.frombuffer(ex["splice"].numpy(), np.uint8)
            s = "".join("ACGT"[i] for i in seq)
            fr, sp = np.where(cod)[0], np.where(spl)[0]
            last_codon = s[fr[-1]:fr[-1] + 3] if len(fr) else ""
            after = s[fr[-1] + 3:fr[-1] + 6] if len(fr) else ""
            recs.append((s, cod.astype(np.float32), spl.astype(np.float32)))
            rows.append(dict(length=L, n_splice=len(sp),
                             splice_at_last_base=bool(len(sp)) and sp[-1] == L - 1,
                             splice_at_first_base=bool(len(sp)) and sp[0] == 0,
                             adjacent_marks=bool(np.any(np.diff(sp) == 1)) if len(sp) > 1 else False,
                             frame_starts_at_0=bool(len(fr)) and fr[0] == 0,
                             cds_starts_ATG=bool(len(fr)) and s[fr[0]:fr[0] + 3] == "ATG",
                             last_frame_codon_is_stop=last_codon in STOPS,
                             codon_after_last_is_stop=after in STOPS,
                             n_frame=len(fr)))
            if len(rows) >= n_max:
                break
        if len(rows) >= n_max:
            break
    d = pd.DataFrame(rows)
    print(f"{len(d):,} records; length median {d.length.median():.0f}, max {d.length.max()}, "
          f"at cap ({SEQ_LEN}) {int((d.length >= SEQ_LEN).sum())}")
    for c in ["splice_at_last_base", "splice_at_first_base", "adjacent_marks", "frame_starts_at_0",
              "cds_starts_ATG", "last_frame_codon_is_stop", "codon_after_last_is_stop"]:
        print(f"  {c:28s} {d[c].mean():.3f}")
    capped = d[d.length >= SEQ_LEN]
    if len(capped):
        print(f"  among capped records: frame channel starts at base 0 / CDS starts with ATG: "
              f"{capped.frame_starts_at_0.mean():.3f} / {capped.cds_starts_ATG.mean():.3f} "
              f"(low ATG fraction = 5' trimming)")
    print("\nHow to read: splice_at_last_base ~0.5 -> marked for one strand only, as in the authors' "
          "code (--splice_encoding saluki_ssd, default); ~1 -> exact; ~0 -> exact_internal; splice_at_first_base ~1 -> exon starts are marked instead "
          "(not supported); last_frame_codon_is_stop ~1 -> CDS includes the stop codon (use "
          "--frame_include_stop); codon_after_last_is_stop ~1 -> stop codon excluded (default here).")
    if fasta and gtf:
        compare_tracks(recs, Fasta(fasta), gtf)


def compare_tracks(recs, fa, gtf):
    """Match training records to GTF transcripts by sequence; which encoding reproduces their tracks?"""
    K = 40
    print("\nMatching records to transcripts (reading GTF and transcript sequences) ...", flush=True)
    txs = {k: v for k, v in read_gtf(gtf).items() if v["cds"] and v["exons"]}
    want = {s[-K:] for s, _, _ in recs if len(s) >= K}
    by_tail = {}
    for tid, t in txs.items():
        try:
            m = mrna_seq(fa, t)
        except KeyError:
            continue
        if m[-K:] in want:
            by_tail.setdefault(m[-K:], []).append((tid, t, m))
    res = []
    for s, cod, spl in recs:
        L = len(s)
        hit = None
        for tid, t, m in by_tail.get(s[-K:], []):
            if (L < SEQ_LEN and m == s) or (L >= SEQ_LEN and m[-SEQ_LEN:] == s):
                hit = (tid, t, m)
                break
        if hit is None:
            continue
        tid, t, m = hit
        r = {"transcript": tid, "strand": t["strand"], "n_exons": len(t["exons"])}
        for mode in SPLICE_MODES:
            r[f"splice_{mode}"] = bool(np.array_equal(splice_track(t, len(m), mode)[-L:], spl))
        for inc in (False, True):
            r[f"frame_stop_{'incl' if inc else 'excl'}"] = bool(np.array_equal(frame_track(t, m, inc)[-L:], cod))
        res.append(r)
    if not res:
        print("No record matched a transcript of this GTF (different annotation release?)")
        return
    d = pd.DataFrame(res)
    print(f"{len(d):,}/{len(recs):,} records matched a transcript exactly by sequence")
    print("Fraction of matched records whose track is reproduced exactly:")
    cols = [c for c in d.columns if c.startswith(("splice_", "frame_"))]
    print(d.groupby("strand")[cols].mean().round(3).to_string())
    print("all  " + "  ".join(f"{c}={d[c].mean():.3f}" for c in cols))
    multi = d[d.n_exons > 1]
    if len(multi):
        print(f"(multi-exon only, n={len(multi)}: " + ", ".join(
            f"{c}={multi[c].mean():.3f}" for c in cols if c.startswith('splice_')) + ")")
    print("Use the --splice_encoding (and --frame_include_stop) setting with the highest fraction.")


# ----------------------------------------------------------------------------- main
def parse_vid(vid):
    chrom, pos, al = vid.split(":")
    ref, alt = al.split(">")
    return chrom, int(pos), ref, alt


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out += hits if hits else ([p] if os.path.exists(p) else [])
    if not out:
        sys.exit(f"No model files match {patterns}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--fasta")
    ap.add_argument("--gtf")
    ap.add_argument("--models", nargs="+", help="Saluki models, e.g. data/external/saluki/train_gru/f*_c*/train/model0_best.h5")
    ap.add_argument("--params", default="data/external/saluki/train_gru/params.json", help="params.json (uses its 'model' block)")
    ap.add_argument("--tx_col", default="TRANSCRIPT_ID")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=200, help="transcripts per write (resume granularity)")
    ap.add_argument("--splice_encoding", choices=SPLICE_MODES, default="saluki_ssd",
                    help="splice channel convention (see above; default reproduces the authors' code)")
    ap.add_argument("--frame_include_stop", action="store_true")
    ap.add_argument("--per_model", action="store_true", help="also write each model's delta")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N variants (testing)")
    ap.add_argument("--inspect_tfr", nargs="+", default=None, help="check encoding conventions and exit")
    ap.add_argument("--inspect_n", type=int, default=2000)
    ap.add_argument("-o", "--output", default=None, help="default: <input>.saluki.tsv")
    a = ap.parse_args()

    if a.inspect_tfr:
        inspect_tfr(a.inspect_tfr, a.inspect_n, a.fasta, a.gtf)
        return
    for req in ("input", "fasta", "gtf", "models"):
        if getattr(a, req) is None:
            ap.error(f"--{req} is required")

    params = dict(DEFAULT_PARAMS)
    if a.params:
        with open(a.params) as fh:
            pj = json.load(fh)
        params.update(pj.get("model", pj))
    if params["seq_length"] != SEQ_LEN:
        sys.exit(f"seq_length {params['seq_length']} != {SEQ_LEN}")
    paths = expand(a.models)

    df = pd.read_csv(a.input, sep="\t", dtype=str, low_memory=False)
    if "variant_id" not in df and "variant_id_x" in df:
        df = df.rename(columns={"variant_id_x": "variant_id"})
    if "variant_id" not in df or a.tx_col not in df:
        sys.exit(f"--input needs variant_id and {a.tx_col}")
    df = df.drop_duplicates("variant_id")[["variant_id", a.tx_col]].copy()
    df["tx"] = df[a.tx_col].astype(str).str.split(".").str[0]
    if a.limit:
        df = df.head(a.limit)

    dest = a.output or re.sub(r"\.tsv(\.gz)?$", "", a.input) + ".saluki.tsv"
    done = set()
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        done = set(pd.read_csv(dest, sep="\t", usecols=["variant_id"], dtype=str)["variant_id"])
        print(f"Resuming: {len(done):,} variants already in {dest}")
    todo = df[~df["variant_id"].isin(done)]
    if todo.empty:
        print("Nothing left to score.")
        return

    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}; GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Loading {len(paths)} model(s) ...", flush=True)
    models = load_models(paths, params)

    fa = Fasta(a.fasta)
    print("Reading GTF ...", flush=True)
    txs = read_gtf(a.gtf, wanted=set(todo["tx"]))
    groups = list(todo.groupby("tx", sort=False))
    print(f"{len(todo):,} variants on {len(groups):,} transcripts "
          f"({sum(1 for tx, _ in groups if tx in txs):,} found in the GTF)", flush=True)

    def predict(arrs):
        """(n_models, n) predictions for a list of SEQ_LEN x 6 arrays."""
        out = np.zeros((len(models), len(arrs)), np.float32)
        for b in range(0, len(arrs), a.batch_size):
            xb = tf.convert_to_tensor(np.stack(arrs[b:b + a.batch_size]))
            for mi, m in enumerate(models):
                out[mi, b:b + len(xb)] = np.asarray(m.predict_on_batch(xb))[:, 0]
        return out

    header = not (os.path.exists(dest) and os.path.getsize(dest) > 0)
    t0, n_scored, n_seq = time.time(), 0, 0
    for c0 in range(0, len(groups), a.chunk):
        rows, seqs = [], []
        for tx, g in groups[c0:c0 + a.chunk]:
            t = txs.get(tx)
            if t is None or not t["exons"]:
                for vid in g["variant_id"]:
                    rows.append({"variant_id": vid, a.tx_col: tx, "saluki_status": "transcript_not_in_gtf"})
                continue
            try:
                mrna, x = encode(fa, t, a.splice_encoding, a.frame_include_stop)
            except KeyError:
                for vid in g["variant_id"]:
                    rows.append({"variant_id": vid, a.tx_col: tx, "saluki_status": "chromosome_not_in_fasta"})
                continue
            xw, off = fit_window(x)
            ref_slot = None
            for vid in g["variant_id"]:
                r = {"variant_id": vid, a.tx_col: tx, "saluki_tx_length": len(mrna), "saluki_trimmed": off > 0}
                try:
                    chrom, pos, ref, alt = parse_vid(vid)
                except ValueError:
                    rows.append({**r, "saluki_status": "bad_variant_id"})
                    continue
                if norm_chrom(chrom) != t["chrom"]:
                    rows.append({**r, "saluki_status": "chromosome_mismatch"})
                    continue
                i = tx_index(t, pos)
                if i is None:
                    rows.append({**r, "saluki_status": "not_exonic"})
                    continue
                r["saluki_tx_pos"] = i + 1
                tref = ref if t["strand"] == "+" else revcomp(ref)
                talt = alt if t["strand"] == "+" else revcomp(alt)
                if len(tref) != 1 or len(talt) != 1 or talt not in NT:
                    rows.append({**r, "saluki_status": "not_snv"})
                    continue
                if mrna[i] != tref:
                    rows.append({**r, "saluki_status": "ref_mismatch"})
                    continue
                if i - off < 0:
                    rows.append({**r, "saluki_status": "outside_window"})
                    continue
                if ref_slot is None:
                    ref_slot = len(seqs)
                    seqs.append(xw)
                xa = xw.copy()
                xa[i - off, :4] = 0
                xa[i - off, NT[talt]] = 1
                rows.append({**r, "saluki_status": "ok", "_ref": ref_slot, "_alt": len(seqs)})
                seqs.append(xa)
        preds = predict(seqs) if seqs else np.zeros((len(models), 0))
        n_seq += len(seqs)
        for r in rows:
            if r.get("saluki_status") == "ok":
                pr, pa = preds[:, r.pop("_ref")], preds[:, r.pop("_alt")]
                dlt = pa - pr
                r.update(saluki_ref=float(pr.mean()), saluki_alt=float(pa.mean()),
                         saluki_delta=float(dlt.mean()),
                         saluki_delta_sd=float(dlt.std(ddof=1)) if len(dlt) > 1 else np.nan,
                         saluki_n_models=len(dlt))
                if a.per_model:
                    r.update({f"saluki_delta_m{k}": float(v) for k, v in enumerate(dlt)})
                n_scored += 1
        cols = ["variant_id", a.tx_col, "saluki_status", "saluki_tx_length", "saluki_tx_pos", "saluki_trimmed",
                "saluki_ref", "saluki_alt", "saluki_delta", "saluki_delta_sd", "saluki_n_models"]
        if a.per_model:
            cols += [f"saluki_delta_m{k}" for k in range(len(models))]
        res = pd.DataFrame(rows).reindex(columns=cols)
        for c in ("saluki_tx_length", "saluki_tx_pos", "saluki_n_models"):
            res[c] = pd.to_numeric(res[c]).astype("Int64")
        res.to_csv(dest, sep="\t", index=False, mode="a", header=header)
        header = False
        el = time.time() - t0
        print(f"  transcripts {min(c0 + a.chunk, len(groups)):,}/{len(groups):,}; scored {n_scored:,}; "
              f"{n_seq / el:.1f} sequences/s across {len(models)} model(s)", flush=True)

    out = pd.read_csv(dest, sep="\t", dtype=str)
    print(f"\nStatus: {out['saluki_status'].value_counts().to_dict()}")
    d = pd.to_numeric(out["saluki_delta"], errors="coerce")
    print(f"saluki_delta: median {d.median():.4f}, IQR {d.quantile(.25):.4f} to {d.quantile(.75):.4f}, "
          f"|delta| 99th pct {d.abs().quantile(.99):.4f}")
    print(f"Wrote {dest} ({len(out):,} variants)")


if __name__ == "__main__":
    main()
