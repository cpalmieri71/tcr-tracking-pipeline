# TCR Tracking Pipeline

This repository contains a set of Python scripts to analyze T-cell receptor (TCR) clonotype
trajectories starting from technical replicates. The workflow includes:

1. **Noise modeling, quality control (QC), and denoising**  
   (`01_noise_pipeline_full_with_freq_v2.py`)

2. **Dynamic modeling** of clonotype frequency trajectories  
   (`02_dynamics_gbm_ou.py`)

3. **Extraction and plotting** of selected clonotype trajectories  
   (`03_extract_plot_clones.py`)

Each script can be run independently depending on the analysis task.

---

## 📂 Repository Structure

TCR_tracking_project/
├── src/
│ ├── noiseK_nb.py # noise model support module
│ └── init.py
├── 01_noise_pipeline_full_with_freq_v2.py
├── 02_dynamics_gbm_ou.py
├── 03_extract_plot_clones.py
├── README.md
├── .gitignore
├── requirements.txt (optional)


---

## 🧠 Requirements

The following Python packages are required:

- **Python 3.9+**
- `numpy`
- `pandas`

Additional script-specific requirements:

- `matplotlib` — needed by `03_extract_plot_clones.py` for plotting
- `scipy` — needed by `02_dynamics_gbm_ou.py` for OU optimization

### Install dependencies

Via pip:

```bash
pip install numpy pandas matplotlib scipy


## Requirements
- Python 3.9+
- requirements.txt
- `noiseK_nb` module available in the Python path

## File: 01_noise_pipeline_full_with_freq_v2.py
This script:
- fits a noise model to technical replicate pairs,
- performs robust QC of model parameters,
- chooses a null pooling strategy,
- performs probabilistic denoising,
- outputs per-clone denoised tables and trajectory observability tables.
- Input expectations

## Input format
TSV files with columns:
- `aaSeqCDR3`
- `readCount`

File naming convention (default):
`<subject>_<time>-<replicate>` where replicate is `1` or `2` (e.g., `1_1-1`, `1_1-2`).

## Example run
```bash
python 01_noise_pipeline_full_with_freq_v2.py \
  --data-dir path/to/replicates \
  --results-dir results \
  --null subject \
  --alpha 0.01 \
  --tail low

## Dynamic modeling of clonotype trajectories

The script `02_dynamics_gbm_ou.py` performs dynamic modeling of clonotype
frequency trajectories derived from the denoised outputs of the noise pipeline.

It compares Geometric Brownian Motion (GBM) and Ornstein–Uhlenbeck (OU)
models in log-frequency space, at the subject level, and reports
model comparison metrics (log-likelihood, AIC, BIC).

Optionally, the OU model can be refined by stratifying transitions
into frequency bins, with bootstrap-based confidence intervals.

### Example
```bash
python 02_dynamics_gbm_ou.py \
  --results-dir results \
  --null subject \
  --alpha 0.01

## Extract and plot clonotype trajectories

Script: `03_extract_plot_clones.py`

This script extracts target and top non-target clonotype frequency trajectories
from one or more long-format trajectory files (e.g.,
`trajectories_long_min2.csv`, `trajectories_long_min5.csv`), and plots
frequency vs time for selected clonotypes.

It supports:
- multiple input files or directories,
- automatic search for files matching `trajectories_long_min*.csv`,
- filtering by mean frequency intervals,
- optional suppression of interactive display (`--no-show`).

### Example
```bash
python 03_extract_plot_clones.py \
  --input trajectories_long_min2.csv trajectories_long_min5.csv \
  --outdir results/

#### Output
selected_clones_by_min_k.csv
multi_clones_extracted_trajectories.csv
multi_clones_observability_by_min_k.csv
multi_clones_frequency_vs_time.png
      |

[![DOI](https://doi.org/10.5281/zenodo.18245016.svg)](https://doi.org/10.5281/zenodo.18245016)
