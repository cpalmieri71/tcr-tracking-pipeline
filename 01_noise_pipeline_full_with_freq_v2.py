#!/usr/bin/env python3
#!/usr/bin/env python3
"""
01_noise_pipeline_full_with_freq_v2.py

End-to-end noise pipeline (script-only; no notebooks).

Overview
--------
1) Fit a noise model for each technical replicate pair (subject_time: replicate 1 vs 2)
2) Robust QC of fitted parameters (MAD-based z-score)
3) Choose a null pooling strategy (pair/subject/global) and compute an audit-only logP cutoff
4) Perform null-consistent probabilistic denoising using an empirical CDF within each pool
5) Aggregate null parameters for reporting/downstream usage
6) Produce "ready-for-trajectories" outputs (clonotype observability over time)

Requirements
------------
- Python 3.9+
- pandas, numpy
- an importable module `noiseK_nb` exposing: `fit_noiseK_nb_powerlaw(...)`
- input replicate files in TSV format with columns: `aaSeqCDR3`, `readCount`

Notes on observability
----------------------
This pipeline defines an empirical p-value from the fitted null distribution:
- p_emp_low  = empirical CDF(logP) within the chosen pool
- p_emp_high = 1 - p_emp_low
The final observability flag is:
    observable <=> (p_value < alpha)
where p_value is chosen from the low or high tail via `--tail`.

The logP cutoff computed in step (3) is retained for auditing/diagnostics only and is NOT
used to define the final `observable` flag.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# --------------------------------------------------
# Allow imports from the local src/ directory
# --------------------------------------------------
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_ROOT))


# -------------------------
# Utilities
# -------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad_val = mad(x)
    return (x - med) / (mad_val + 1e-12)


def empirical_cdf(sorted_arr: np.ndarray, x: float) -> float:
    """Return P(X <= x) given a sorted array."""
    if sorted_arr.size == 0:
        return float("nan")
    return float(np.searchsorted(sorted_arr, x, side="right") / sorted_arr.size)


# -------------------------
# Noise model fitting
# -------------------------
PATTERN_DEFAULT = r"^(?P<subject>\d+)_(?P<time>\d+)-(?P<replica>[12])$"


def read_rep_file(fp: Path, sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(fp, sep=sep)
    required = {"aaSeqCDR3", "readCount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{fp.name}: missing columns: {missing}. Expected: {sorted(required)}"
        )
    return df[["aaSeqCDR3", "readCount"]].copy()


def merge_two_reps(
    fp1: Path, fp2: Path, sep: str = "\t", key: str = "aaSeqCDR3"
) -> Tuple[pd.DataFrame, List[int], List[str], int, int]:
    d1 = read_rep_file(fp1, sep=sep).rename(columns={"readCount": "count_rep1"})
    d2 = read_rep_file(fp2, sep=sep).rename(columns={"readCount": "count_rep2"})

    merged = d1.merge(d2, on=key, how="outer")
    merged["count_rep1"] = merged["count_rep1"].fillna(0).astype(int)
    merged["count_rep2"] = merged["count_rep2"].fillna(0).astype(int)

    depth_rep1 = int(merged["count_rep1"].sum())
    depth_rep2 = int(merged["count_rep2"].sum())

    depths = [depth_rep1, depth_rep2]
    count_cols = ["count_rep1", "count_rep2"]
    return merged, depths, count_cols, depth_rep1, depth_rep2


def index_pairs(data_dir: Path, pattern: re.Pattern) -> List[Tuple[int, int, Path, Path]]:
    all_files = [fp for fp in data_dir.iterdir() if fp.is_file()]
    index: Dict[Tuple[int, int, int], Path] = {}

    for fp in all_files:
        m = pattern.match(fp.name)
        if not m:
            continue
        subject = int(m.group("subject"))
        time = int(m.group("time"))
        replica = int(m.group("replica"))
        index[(subject, time, replica)] = fp

    # Auto-discover complete pairs (replicate 1 & 2)
    pairs: List[Tuple[int, int, Path, Path]] = []
    seen_st = sorted({(s, t) for (s, t, _) in index.keys()})
    for subject, time in seen_st:
        fp1 = index.get((subject, time, 1))
        fp2 = index.get((subject, time, 2))
        if fp1 and fp2:
            pairs.append((subject, time, fp1, fp2))
    return pairs


def run_noise_fit(
    data_dir: Path,
    results_dir: Path,
    perclone_dir: Path,
    file_sep: str,
    pattern_str: str,
    gamma_init: float,
    k_init: float,
    grid_size: int,
) -> Path:
    """
    Fit the noise model for each (subject,time) replicate pair and save:
    - results_dir/null_params_by_pair_std.csv
    - perclone_dir/null_per_clone_logP_std{subject_time}.csv
    """
    try:
        from noiseK_nb import fit_noiseK_nb_powerlaw
    except Exception as e:
        raise ImportError(
            "Cannot import noiseK_nb.fit_noiseK_nb_powerlaw. "
            "Make sure noiseK_nb.py (or the package) is on PYTHONPATH or in the same folder."
        ) from e

    ensure_dir(results_dir)
    ensure_dir(perclone_dir)

    pattern = re.compile(pattern_str)
    pairs = index_pairs(data_dir, pattern)
    print(f"Found {len(pairs)} complete pairs (replicate 1 & 2) in {data_dir}")

    rows = []
    for subject, time, fp1, fp2 in pairs:
        pair_id = f"{subject}_{time}"
        print(f"\n--- Processing {pair_id}: {fp1.name} + {fp2.name}")

        try:
            merged, depths, count_cols, depth_rep1, depth_rep2 = merge_two_reps(
                fp1, fp2, sep=file_sep
            )

            # Frequencies for downstream dynamics
            merged["freq_rep1"] = merged["count_rep1"] / (depth_rep1 if depth_rep1 > 0 else 1)
            merged["freq_rep2"] = merged["count_rep2"] / (depth_rep2 if depth_rep2 > 0 else 1)
            merged["freq_geo"] = np.sqrt(merged["freq_rep1"] * merged["freq_rep2"])

            fit = fit_noiseK_nb_powerlaw(
                merged,
                depths,      # [depth1, depth2]
                count_cols,  # ["count_rep1","count_rep2"]
                gamma_init=gamma_init,
                k_init=k_init,
                grid_size=grid_size,
            )

            # Save per-clone logP
            perclone_out = perclone_dir / f"null_per_clone_logP_std{pair_id}.csv"

            # Enrich per-clone logP with counts/frequencies
            per_clone = fit.per_clone_logP.copy()

            if "aaSeqCDR3" in per_clone.columns:
                key_col = "aaSeqCDR3"
            elif "cloneId" in per_clone.columns:
                key_col = "cloneId"
            elif "cloneID" in per_clone.columns:
                key_col = "cloneID"
            else:
                # Fallback: first column as key (less safe, but avoids crashing)
                key_col = per_clone.columns[0]

            extra_cols = ["count_rep1", "count_rep2", "freq_rep1", "freq_rep2", "freq_geo"]
            extra = (
                merged[[key_col] + [c for c in extra_cols if c in merged.columns]].copy()
                if key_col in merged.columns
                else None
            )
            if extra is not None:
                per_clone = per_clone.merge(extra, on=key_col, how="left")

            # Add depths as constants (useful for auditing)
            per_clone["depth_rep1"] = depth_rep1
            per_clone["depth_rep2"] = depth_rep2

            per_clone.to_csv(perclone_out, index=False)

            # Summary row
            row = {
                "subject": subject,
                "time": time,
                "file_rep1": fp1.name,
                "file_rep2": fp2.name,
                "depth_rep1": depth_rep1,
                "depth_rep2": depth_rep2,
                "success": bool(getattr(fit, "success", True)),
                "message": str(getattr(fit, "message", "")),
                "log_likelihood": float(getattr(fit, "log_likelihood", np.nan)),
            }

            # Include all fit parameters if available
            params = getattr(fit, "params", {})
            if isinstance(params, dict):
                for k, v in params.items():
                    row[k] = v

            rows.append(row)

        except Exception as e:
            eprint(f"ERROR in {pair_id}: {e}")
            rows.append(
                {
                    "subject": subject,
                    "time": time,
                    "file_rep1": fp1.name,
                    "file_rep2": fp2.name,
                    "depth_rep1": None,
                    "depth_rep2": None,
                    "success": False,
                    "message": f"ERROR: {e}",
                }
            )

    out_table = results_dir / "null_params_by_pair_std.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)
    print(f"\nSaved parameter table: {out_table}")
    print(f"Saved per-clone logP files in: {perclone_dir}")
    return out_table


# -------------------------
# QC
# -------------------------
def run_qc(
    null_params_csv: Path,
    qc_out_csv: Path,
    mad_cutoff: float = 3.5,
) -> pd.DataFrame:
    df = pd.read_csv(null_params_csv)

    # Use successful fits only
    df_qc = df[df.get("success", False) == True].copy()
    if df_qc.empty:
        # Still save a QC file with all failures
        df_fail = df.copy()
        df_fail["QC_pass"] = False
        df_fail["QC_reason"] = "fit_failed_or_missing"
        ensure_dir(qc_out_csv.parent)
        df_fail.to_csv(qc_out_csv, index=False)
        return df_fail

    # Expected columns from the fit
    for col in ["gamma", "k", "log_likelihood"]:
        if col not in df_qc.columns:
            raise ValueError(
                f"QC: missing column '{col}' in {null_params_csv}. Columns: {list(df_qc.columns)}"
            )

    df_qc["z_gamma"] = robust_zscore(df_qc["gamma"].to_numpy(dtype=float))
    df_qc["z_k"] = robust_zscore(df_qc["k"].to_numpy(dtype=float))
    df_qc["z_ll"] = robust_zscore(df_qc["log_likelihood"].to_numpy(dtype=float))

    df_qc["gamma_outlier"] = df_qc["z_gamma"].abs() > mad_cutoff
    df_qc["k_outlier"] = df_qc["z_k"].abs() > mad_cutoff
    df_qc["ll_outlier"] = df_qc["z_ll"].abs() > mad_cutoff

    df_qc["QC_pass"] = ~(df_qc["gamma_outlier"] | df_qc["k_outlier"] | df_qc["ll_outlier"])
    df_qc["QC_reason"] = np.where(df_qc["QC_pass"], "pass", "parameter_outlier")

    # Add failures back
    df_fail = df[df.get("success", False) == False].copy()
    if not df_fail.empty:
        df_fail["QC_pass"] = False
        df_fail["QC_reason"] = "fit_failed"

    df_final = pd.concat([df_qc, df_fail], ignore_index=True)

    ensure_dir(qc_out_csv.parent)
    df_final.to_csv(qc_out_csv, index=False)
    print("QC completed.")
    print(df_final["QC_pass"].value_counts(dropna=False))
    return df_final


# -------------------------
# Null choice + denoising + aggregations
# -------------------------
def infer_clone_and_logp_cols(perclone: pd.DataFrame) -> Tuple[str, str]:
    clone_col_candidates = ["aaSeqCDR3", "cloneId", "cloneID", "clonotype", "cdr3"]
    logp_col_candidates = ["logP", "logp", "log_prob", "log_probability", "logP_noise", "logPnull"]

    clone_col = next((c for c in clone_col_candidates if c in perclone.columns), None)
    if clone_col is None:
        raise ValueError(
            f"Cannot find a clonotype column among {clone_col_candidates}. Columns: {list(perclone.columns)}"
        )

    logp_col = next((c for c in logp_col_candidates if c in perclone.columns), None)
    if logp_col is None:
        raise ValueError(
            f"Cannot find a logP column among {logp_col_candidates}. Columns: {list(perclone.columns)}"
        )

    return clone_col, logp_col


def load_perclone_with_metadata(perclone_dir: Path, df_qc: pd.DataFrame) -> pd.DataFrame:
    files = sorted(perclone_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {perclone_dir}")

    # QC map for (subject,time)
    if not {"subject", "time", "QC_pass"}.issubset(df_qc.columns):
        raise ValueError("df_qc must contain: subject, time, QC_pass")

    qc_key = df_qc.set_index(["subject", "time"])["QC_pass"].to_dict()

    rows = []
    pair_pat = re.compile(r"(?P<sid>\d+_\d+)")  # extracts subject_time from any stem containing \d+_\d+
    for fp in files:
        m = pair_pat.search(fp.stem)
        if not m:
            continue
        pair_id = m.group("sid")
        subject, time = map(int, pair_id.split("_"))

        d = pd.read_csv(fp)
        d["subject"] = subject
        d["time"] = time
        d["pair_id"] = pair_id
        d["QC_pass_pair"] = bool(qc_key.get((subject, time), False))
        rows.append(d)

    if not rows:
        raise ValueError(f"Could not parse subject_time from filenames in {perclone_dir}")

    return pd.concat(rows, ignore_index=True)


def apply_pool_and_cutoff(
    perclone: pd.DataFrame,
    df_qc: pd.DataFrame,
    choice: str,
    alpha: float,
    min_qc_pairs_per_subject: int,
    logp_col: str,
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pool_id and an audit-only logP cutoff per pool from QC-pass pairs.

    IMPORTANT: The cutoff-based flag is kept for diagnostics only and is NOT used
    to define final observability.
    """
    clean = perclone[perclone["QC_pass_pair"] == True].copy()
    if clean.empty:
        raise ValueError("No QC-pass pairs available to build the null pool.")

    # pool_id according to strategy
    if choice == "clone":
        clean["pool_id"] = clean["pair_id"]
        perclone["pool_id"] = perclone["pair_id"]
    elif choice == "subject":
        clean["pool_id"] = clean["subject"].astype(str)
        perclone["pool_id"] = perclone["subject"].astype(str)
    elif choice == "global":
        clean["pool_id"] = "GLOBAL"
        perclone["pool_id"] = "GLOBAL"
    else:
        raise ValueError("choice must be one of: clone/subject/global")

    # subject -> global fallback if too few QC-pass pairs
    if choice == "subject":
        qc_pass_counts = (
            df_qc[df_qc["QC_pass"] == True].groupby("subject").size().rename("n_qc_pairs").to_dict()
        )
        perclone["n_qc_pairs_subject"] = perclone["subject"].map(qc_pass_counts).fillna(0).astype(int)
        clean["n_qc_pairs_subject"] = clean["subject"].map(qc_pass_counts).fillna(0).astype(int)

        perclone["pool_id"] = np.where(
            perclone["n_qc_pairs_subject"] >= min_qc_pairs_per_subject,
            perclone["subject"].astype(str),
            "GLOBAL",
        )
        clean["pool_id"] = np.where(
            clean["n_qc_pairs_subject"] >= min_qc_pairs_per_subject,
            clean["subject"].astype(str),
            "GLOBAL",
        )

    # cutoff per pool (audit only)
    cutoffs = clean.groupby("pool_id")[logp_col].quantile(alpha).rename("logP_cutoff").reset_index()
    cutoff_map = dict(zip(cutoffs["pool_id"], cutoffs["logP_cutoff"]))

    perclone["logP_cutoff"] = perclone["pool_id"].map(cutoff_map)

    # global fallback if a pool is missing
    if choice != "global":
        global_cut = float(clean[logp_col].quantile(alpha))
        perclone["logP_cutoff"] = perclone["logP_cutoff"].fillna(global_cut)

    perclone["observable_cutoff"] = perclone[logp_col] >= perclone["logP_cutoff"]

    ensure_dir(results_dir)
    cutoffs_out = results_dir / f"logP_cutoffs_{choice}_alpha{alpha}.csv"
    cutoffs.to_csv(cutoffs_out, index=False)
    print("Saved cutoffs:", cutoffs_out)

    return perclone, cutoffs


def denoise_empirical(perclone: pd.DataFrame, logp_col: str) -> pd.DataFrame:
    """
    Compute an empirical CDF-based p-value proxy within each pool_id,
    using QC-pass pairs only to define the reference distributions.
    """
    clean = perclone[perclone["QC_pass_pair"] == True].copy()
    if clean.empty:
        raise ValueError("No QC-pass pairs available: cannot compute empirical CDFs.")

    pool_arrays: Dict[str, np.ndarray] = {
        pid: np.sort(g[logp_col].to_numpy(dtype=float)) for pid, g in clean.groupby("pool_id")
    }

    if "GLOBAL" not in pool_arrays:
        pool_arrays["GLOBAL"] = np.sort(clean[logp_col].to_numpy(dtype=float))

    p_emp_low: List[float] = []
    for pid, x in zip(perclone["pool_id"].astype(str).to_numpy(), perclone[logp_col].to_numpy(dtype=float)):
        arr = pool_arrays.get(pid, pool_arrays["GLOBAL"])
        p_emp_low.append(empirical_cdf(arr, float(x)))

    perclone["p_emp_low"] = p_emp_low
    return perclone


def aggregate_null_params(df_qc: pd.DataFrame, choice: str, results_dir: Path) -> Optional[Path]:
    params_clean = df_qc[df_qc["QC_pass"] == True].copy()
    if params_clean.empty:
        raise ValueError("No QC-pass pairs available: cannot aggregate parameters.")

    candidates = ["gamma", "fmin", "k", "K", "N_total_mean", "N_total_min", "N_total_max"]
    cols = [c for c in candidates if c in params_clean.columns]

    if not cols:
        print("Note: no aggregatable parameter columns found (beyond subject/time). Skipping aggregation.")
        return None

    ensure_dir(results_dir)

    if choice == "subject":
        agg = params_clean.groupby("subject")[cols].median(numeric_only=True).reset_index()
        out = results_dir / "null_params_by_subject_median.csv"
        agg.to_csv(out, index=False)
        print("Saved subject-level aggregated parameters:", out)
        return out

    if choice == "global":
        med = params_clean[cols].median(numeric_only=True)
        agg = pd.DataFrame([med])
        agg.insert(0, "pool_id", "GLOBAL")
        out = results_dir / "null_params_global_median.csv"
        agg.to_csv(out, index=False)
        print("Saved global aggregated parameters:", out)
        return out

    # choice == clone
    print("choice=clone: using pair-level parameters already saved in null_params_by_pair_std.csv")
    return None


def trajectory_outputs(
    perclone: pd.DataFrame, clone_col: str, choice: str, alpha: float, results_dir: Path
) -> Tuple[Path, Path]:
    """
    Produce two outputs:
    - per-timepoint observability per clonotype
    - per-clonotype counts across time
    """
    traj = (
        perclone.groupby(["subject", clone_col, "time"], as_index=False)
        .agg(observable=("observable", "max"), p_emp_low=("p_emp_low", "min"))
    )

    traj_counts = (
        traj.groupby(["subject", clone_col], as_index=False)
        .agg(n_times_observable=("observable", "sum"), n_times_total=("time", "nunique"))
    )

    out1 = results_dir / f"trajectory_observability_{choice}_alpha{alpha}.csv"
    out2 = results_dir / f"trajectory_observability_counts_{choice}_alpha{alpha}.csv"
    traj.to_csv(out1, index=False)
    traj_counts.to_csv(out2, index=False)

    print("Saved trajectory-ready outputs:")
    print(" -", out1)
    print(" -", out2)
    return out1, out2


# -------------------------
# Main
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: noise fit -> QC -> null pooling -> denoising -> aggregation -> trajectory outputs"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with replicate files (e.g., repertories_longitudinal) named like '1_1-1' and '1_1-2'",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Main output folder (default: ./results)",
    )
    p.add_argument(
        "--file-sep",
        type=str,
        default="\t",
        help="Input file separator (default: tab for .tsv)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=PATTERN_DEFAULT,
        help=f"Regex for replicate filenames (default: {PATTERN_DEFAULT})",
    )

    # Fit params
    p.add_argument("--gamma-init", type=float, default=1.6)
    p.add_argument("--k-init", type=float, default=50.0)
    p.add_argument("--grid-size", type=int, default=500)

    # QC
    p.add_argument("--mad-cutoff", type=float, default=3.5)

    # Null choice + alpha
    p.add_argument(
        "--null",
        dest="null_choice",
        choices=["clone", "subject", "global"],
        required=True,
        help="Null pooling strategy: clone (=per pair), subject (=per subject pool), global (=single global pool)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Alpha quantile / p-value threshold (default: 0.01)",
    )
    p.add_argument(
        "--min-qc-pairs-per-subject",
        type=int,
        default=3,
        help="If null=subject, minimum QC-pass pairs required to use subject pool (otherwise GLOBAL).",
    )

    # Tail selection for p-value definition
    p.add_argument(
        "--tail",
        choices=["low", "high"],
        default="low",
        help=(
            "Which tail defines rare events under the null (default: low). "
            "low -> p_value = p_emp_low (CDF), high -> p_value = 1 - p_emp_low. "
            "Final observable is defined as (p_value < alpha)."
        ),
    )

    # Advanced: skip fit/qc if already computed
    p.add_argument(
        "--skip-fit",
        action="store_true",
        help="Skip fitting and use an existing results-dir/null_params_by_pair_std.csv",
    )
    p.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip QC and use an existing results-dir/QC_noise_model_results_std.csv",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    data_dir: Path = args.data_dir
    results_dir: Path = args.results_dir
    perclone_dir: Path = results_dir / "per_clone_long_logP"

    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    # 1) Fit
    null_params_csv = results_dir / "null_params_by_pair_std.csv"
    if args.skip_fit:
        if not null_params_csv.exists():
            raise FileNotFoundError(f"--skip-fit was requested but file not found: {null_params_csv}")
        print(f"[skip-fit] Using {null_params_csv}")
    else:
        null_params_csv = run_noise_fit(
            data_dir=data_dir,
            results_dir=results_dir,
            perclone_dir=perclone_dir,
            file_sep=args.file_sep,
            pattern_str=args.pattern,
            gamma_init=args.gamma_init,
            k_init=args.k_init,
            grid_size=args.grid_size,
        )

    # 2) QC
    qc_out_csv = results_dir / "QC_noise_model_results_std.csv"
    if args.skip_qc:
        if not qc_out_csv.exists():
            raise FileNotFoundError(f"--skip-qc was requested but file not found: {qc_out_csv}")
        df_qc = pd.read_csv(qc_out_csv)
        print(f"[skip-qc] Using {qc_out_csv}")
    else:
        df_qc = run_qc(
            null_params_csv=null_params_csv,
            qc_out_csv=qc_out_csv,
            mad_cutoff=args.mad_cutoff,
        )

    # 3) Load per-clone + metadata
    perclone = load_perclone_with_metadata(perclone_dir=perclone_dir, df_qc=df_qc)
    print("Total per-clone rows:", len(perclone))

    clone_col, logp_col = infer_clone_and_logp_cols(perclone)
    print("Using clone_col =", clone_col, "| logp_col =", logp_col)

    # 4) Pool definition + audit cutoff (does not define final observability)
    perclone, _cutoffs = apply_pool_and_cutoff(
        perclone=perclone,
        df_qc=df_qc,
        choice=args.null_choice,
        alpha=float(args.alpha),
        min_qc_pairs_per_subject=int(args.min_qc_pairs_per_subject),
        logp_col=logp_col,
        results_dir=results_dir,
    )

    # 5) Empirical denoising consistent with pool choice
    perclone = denoise_empirical(perclone=perclone, logp_col=logp_col)

    # 6) Define p_value + final observability in a self-consistent way
    perclone["p_emp_low"] = pd.to_numeric(perclone["p_emp_low"], errors="raise")
    perclone["p_emp_high"] = 1.0 - perclone["p_emp_low"]

    if args.tail == "low":
        perclone["p_value"] = perclone["p_emp_low"]
    else:
        perclone["p_value"] = perclone["p_emp_high"]

    perclone["observable"] = perclone["p_value"] < float(args.alpha)

    # Safety assertion: prevent incoherent states
    bad = perclone[(perclone["p_value"] >= float(args.alpha)) & (perclone["observable"] == True)]
    if len(bad) > 0:
        cols = [
            c
            for c in ["subject", "time", clone_col, logp_col, "p_emp_low", "p_emp_high", "p_value", "observable"]
            if c in bad.columns
        ]
        raise RuntimeError(
            f"Incoherent observability detected: {len(bad)} rows with p_value>=alpha but observable=True.\n"
            f"Examples:\n{bad[cols].head(5).to_string(index=False)}"
        )

    den_out = results_dir / f"per_clone_denoised_{args.null_choice}_alpha{args.alpha}.csv"
    perclone.to_csv(den_out, index=False)
    print("Saved per-clone denoised table:", den_out)

    # 7) Aggregate null parameters (reporting/downstream)
    aggregate_null_params(df_qc=df_qc, choice=args.null_choice, results_dir=results_dir)

    # 8) Trajectory-ready outputs
    trajectory_outputs(
        perclone=perclone,
        clone_col=clone_col,
        choice=args.null_choice,
        alpha=float(args.alpha),
        results_dir=results_dir,
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
