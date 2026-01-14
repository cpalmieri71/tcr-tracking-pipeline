# Noise pipeline (technical replicates)

This repository contains an end-to-end Python script to fit a noise model on technical replicate pairs,
perform robust QC, define a null pooling strategy, compute empirical p-values, and export clonotype
observability tables ready for longitudinal trajectory analyses.

## Requirements
- Python 3.9+
- numpy, pandas
- `noiseK_nb` module available in the Python path

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
