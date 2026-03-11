# CSM Batch Processor

**Cross-Spectral Matrix calculator and spectral analysis toolkit**

Computes narrowband cross-spectral matrices from multi-channel time-series data using Welch's block-averaging method with Hanning windowing.

---

## Features

| Feature | Description |
|---|---|
| **Narrowband CSM** | Welch-style block-averaged cross-spectral matrix with configurable overlap and record length |
| **Octave-band binning** | Fractional-octave frequency binning (1/3, 1/12, etc.) via `logfnan` algorithm |
| **Correlation** | Normalised auto- and cross-correlation for all channel pairs |
| **Visualisation** | Publication-quality spectral, coherence, and correlation plots |
| **Multi-format I/O** | Reads CSV, MATLAB `.mat` (v5–v7.3), and NI TDMS files |
| **Batch CLI** | Process multiple files from the command line |
| **React GUI** | Interactive browser-based demo for quick analysis |

## Installation

```bash
pip install -e .

# With TDMS support
pip install -e ".[tdms]"

# Everything
pip install -e ".[all]"
```

## Quick Start — Python API

```python
import numpy as np
from csm_processor import csm_calculator, load_data, plot_autospectra

# Load data (CSV, MAT, or TDMS)
data = load_data("experiment_001.csv")

# Compute cross-spectral matrix
spectra, freq = csm_calculator(data, fs=51200, n_rec=4096, overlap=50)

# Plot auto-spectra
fig = plot_autospectra(freq, spectra, title="Wind Tunnel Run 001")
fig.savefig("autospectra.png", dpi=200)
```

### Octave-Band Binning

```python
from csm_processor.log_binning import bin_csm

df = 51200 / 4096  # = 12.5 Hz
freq_binned, spectra_binned = bin_csm(df, spectra, bins_per_octave=3)
```

### Correlation

```python
from csm_processor import compute_correlation, plot_correlation

tau, corr_matrix = compute_correlation(data, fs=51200)
fig = plot_correlation(tau, corr_matrix)
```

### Coherence

```python
from csm_processor import plot_coherence

fig = plot_coherence(freq, spectra, ch_i=0, ch_j=1)
```

## Command-Line Interface

```bash
# Single file
python -m csm_processor data.csv --fs 51200 --nrec 4096 --overlap 50 --plot

# Batch with binning
python -m csm_processor *.tdms --fs 51200 --nrec 4096 \
    --bin --bpo 3 --correlation --plot --outdir results/

# Save as .mat (MATLAB-compatible)
python -m csm_processor data.csv --fs 51200 --nrec 4096 --fmt mat
```

## Architecture

```
csm_processor/
├── __init__.py          # Public API
├── __main__.py          # python -m entry
├── cli.py               # Batch CLI
├── csm_calculator.py    # Core CSM engine (port of CSM_Calculator.m)
├── log_binning.py       # Fractional-octave binning (port of logfnan.m)
├── correlation.py       # Auto/cross-correlation
├── io_utils.py          # CSV / MAT / TDMS loaders + save helpers
└── plotting.py          # Matplotlib visualisation
```

### CSM Computation — How It Works

The algorithm follows Welch's method:

1. **Segment** the time-series into overlapping blocks of length `n_rec`
2. **Window** each block with a periodic Hanning window (energy-normalised by √0.375)
3. **FFT** each windowed block per channel
4. **Outer product** at each frequency: `CSM[f,i,j] = S[f,i] × conj(S[f,j])`
5. **Average** across all blocks and normalise: `CSM = 2·Σ / (n_rec · fs · n_blocks)`

This is a direct port of the original MATLAB `CSM_Calculator.m` function.

## MATLAB Equivalence

| MATLAB (original) | Python (this package) |
|---|---|
| `CSM_Calculator(data, M, fs, N, Nrec, overlap)` | `csm_calculator(data, fs, n_rec, overlap)` |
| `logfnan(df, g, d)` | `log_freq_bin(df, spectrum, bins_per_octave)` |
| `xcorr(x, y, 'normalized')` | `compute_correlation(data, fs)` |
| `spectra(:,i,j)` | `spectra[:, i, j]` |

## Output Format

Results are saved as `.npz` (default) or `.mat` files containing:

- `spectra` — Cross-spectral matrix `(N_freq × M × M)`
- `freq` — Frequency vector `(N_freq,)`
- `fs`, `n_rec`, `overlap` — Processing parameters
- `freq_binned`, `spectra_binned` — (if `--bin` used)
- `tau`, `corr_matrix` — (if `--correlation` used)

## License

MIT
