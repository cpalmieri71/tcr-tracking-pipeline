# noiseK_nb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp


@dataclass
class FitResult:
    params: Dict[str, float]
    log_likelihood: float
    success: bool
    message: str
    nfev: int
    per_clone_logP: pd.DataFrame


def load_and_merge_K(
    files: List[str],
    key: str = "aaSeqCDR3",
    count_col: str = "readCount",
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    """
    Read K CSV files and outer-merge on `key`, filling missing counts with 0.
    Returns:
      merged df with columns: key, n1..nK
      depths: [N1..NK]
      colnames: ["n1".. "nK"]
    """
    if len(files) < 2:
        raise ValueError("Need at least 2 files to estimate noise.")

    dfs = []
    depths = []
    colnames = []

    for i, fp in enumerate(files, start=1):
        df = pd.read_csv(fp)
        if key not in df.columns or count_col not in df.columns:
            raise ValueError(f"File {fp} missing '{key}' or '{count_col}'. Columns={list(df.columns)}")
        Ni = int(df[count_col].sum())
        if Ni <= 0:
            raise ValueError(f"File {fp} has non-positive depth (sum {count_col}={Ni}).")
        depths.append(Ni)

        col = f"n{i}"
        colnames.append(col)
        dfs.append(df[[key, count_col]].rename(columns={count_col: col}))

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=key, how="outer")

    merged[colnames] = merged[colnames].fillna(0).astype(int)

    return merged, depths, colnames


def _log_prior_powerlaw(f: np.ndarray, gamma: float, fmin: float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if not (gamma > 0 and 0 < fmin < 1):
        return np.full_like(f, -np.inf, dtype=float)

    if np.isclose(gamma, 1.0):
        logC = -np.log(np.log(1.0 / fmin))
    else:
        num = (1.0 - gamma)
        den = (1.0 - fmin ** (1.0 - gamma))
        logC = np.log(abs(num)) - np.log(abs(den))

    return logC - gamma * np.log(f)


def _log_nb_pmf(n: np.ndarray, mu: np.ndarray, k: float) -> np.ndarray:
    """
    NB log-PMF (broadcast-safe), mean=mu, dispersion=k:
      Var = mu + mu^2/k
    """
    if k <= 0:
        return np.full(np.broadcast(n, mu).shape, -np.inf, dtype=float)

    n = np.asarray(n, dtype=float)
    mu = np.asarray(mu, dtype=float)

    log_coeff = gammaln(n + k) - gammaln(k) - gammaln(n + 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_k_over = np.log(k) - np.log(k + mu)     # log(k/(k+mu))
        log_mu_over = np.log(mu) - np.log(k + mu)   # log(mu/(k+mu))

    logp = log_coeff + k * log_k_over + n * log_mu_over
    logp = np.where(mu > 0, logp, np.where(n == 0, 0.0, -np.inf))
    return logp


def per_clone_logP_nb_powerlaw_K(
    counts: np.ndarray,          # shape (I, K)
    depths: List[int],           # len K
    gamma: float,
    fmin: float,
    k: float,
    grid_size: int = 500,
) -> np.ndarray:
    """
    For each clone i:
      log ∫ rho(f) Π_r NB(n_ir | mu=N_r f, k) df
    """
    I, Krep = counts.shape
    if Krep != len(depths):
        raise ValueError("counts.shape[1] must match len(depths)")

    f = np.logspace(np.log10(fmin), 0.0, int(grid_size))
    # trapezoid weights
    w = np.zeros_like(f)
    w[1:-1] = 0.5 * (f[2:] - f[:-2])
    w[0] = 0.5 * (f[1] - f[0])
    w[-1] = 0.5 * (f[-1] - f[-2])
    if np.any(w <= 0):
        raise RuntimeError("Non-positive integration weights.")
    logw = np.log(w)

    logprior = _log_prior_powerlaw(f, gamma=gamma, fmin=fmin)  # (G,)

    # term(i,g) = logprior(g)+logw(g) + Σ_r log NB(n_ir | mu=N_r f_g, k)
    term = (logprior[None, :] + logw[None, :]).copy()  # (1,G) broadcast to (I,G)

    for r, Nr in enumerate(depths):
        mu = Nr * f  # (G,)
        term = term + _log_nb_pmf(counts[:, r][:, None], mu[None, :], k)

    return logsumexp(term, axis=1)  # (I,)


def fit_noiseK_nb_powerlaw(
    merged: pd.DataFrame,
    depths: List[int],
    count_cols: List[str],
    gamma_init: float = 1.6,
    fmin_init: Optional[float] = None,
    k_init: float = 50.0,
    gamma_bounds: Tuple[float, float] = (1.01, 6.0),
    fmin_bounds: Optional[Tuple[float, float]] = None,
    k_bounds: Tuple[float, float] = (1e-2, 1e6),
    grid_size: int = 500,
) -> FitResult:
    counts = merged[count_cols].to_numpy(dtype=float)  # (I,K)
    depths = list(map(int, depths))

    maxN = max(depths)
    if fmin_bounds is None:
        fmin_bounds = (1.0 / (10.0 * maxN), 1e-2)
    if fmin_init is None:
        fmin_init = max(fmin_bounds[0], 1.0 / (3.0 * maxN))

    x0 = np.array([gamma_init, fmin_init, k_init], dtype=float)

    def objective(x: np.ndarray) -> float:
        gamma, fmin, k = map(float, x)
        if not (gamma_bounds[0] <= gamma <= gamma_bounds[1]):
            return 1e50
        if not (fmin_bounds[0] <= fmin <= fmin_bounds[1]):
            return 1e50
        if not (k_bounds[0] <= k <= k_bounds[1]):
            return 1e50

        logP = per_clone_logP_nb_powerlaw_K(
            counts=counts, depths=depths, gamma=gamma, fmin=fmin, k=k, grid_size=grid_size
        )
        nll = -float(np.sum(logP))
        return nll if np.isfinite(nll) else 1e50

    res = minimize(objective, x0, method="L-BFGS-B", bounds=[gamma_bounds, fmin_bounds, k_bounds])

    gamma_hat, fmin_hat, k_hat = map(float, res.x)
    logP_hat = per_clone_logP_nb_powerlaw_K(
        counts=counts, depths=depths, gamma=gamma_hat, fmin=fmin_hat, k=k_hat, grid_size=grid_size
    )
    ll = float(np.sum(logP_hat))

    out = merged.copy()
    out["logP"] = logP_hat
    out["total_counts"] = out[count_cols].sum(axis=1)

    return FitResult(
        params={
            "gamma": gamma_hat,
            "fmin": fmin_hat,
            "k": k_hat,
            "K": float(len(depths)),
            "N_total_mean": float(np.mean(depths)),
            "N_total_min": float(np.min(depths)),
            "N_total_max": float(np.max(depths)),
        },
        log_likelihood=ll,
        success=bool(res.success),
        message=str(res.message),
        nfev=int(getattr(res, "nfev", -1)),
        per_clone_logP=out,
    )


def make_diagnostic_plots(
    fit: FitResult,
    outdir: str = "figures",
    bins: int = 60,
):
    os.makedirs(outdir, exist_ok=True)
    df = fit.per_clone_logP

    plt.figure()
    plt.hist(df["logP"], bins=bins)
    plt.xlabel("log P({n_r} | model)")
    plt.ylabel("Number of clonotypes")
    plt.title("Per-clonotype log-likelihood (NoiseK-NB)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "logP_hist_NoiseK_NB.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(df["total_counts"], df["logP"], s=8, alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Total counts across replicates")
    plt.ylabel("log P({n_r} | model)")
    plt.title("Log-likelihood vs abundance (NoiseK-NB)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "logP_vs_total_counts_NoiseK_NB.png"), dpi=300)
    plt.close()


def aic(log_likelihood: float, n_params: int) -> float:
    return 2.0 * float(n_params) - 2.0 * float(log_likelihood)
