#!/usr/bin/env python3
"""
CSM Batch Processor — Command-Line Interface
==============================================
Process one or more data files from the command line.

Usage examples
--------------
    # Single file
    python -m csm_processor.cli data.csv --fs 51200 --nrec 4096 --overlap 50

    # Batch (glob)
    python -m csm_processor.cli *.tdms --fs 51200 --nrec 4096 --overlap 50 --outdir results/

    # With binning and correlation
    python -m csm_processor.cli data.mat --fs 51200 --nrec 4096 --overlap 50 \
        --bin --bpo 3 --correlation --plot
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

from .csm_calculator import csm_calculator
from .log_binning import bin_csm
from .correlation import compute_correlation
from .io_utils import load_data, save_results
from .plotting import plot_autospectra, plot_coherence, plot_correlation


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="csm_processor",
        description="Cross-Spectral Matrix Batch Processor",
    )
    p.add_argument("files", nargs="+", help="Input file(s) — CSV, MAT, or TDMS")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    p.add_argument("--nrec", type=int, required=True, help="Record length (samples per block)")
    p.add_argument("--overlap", type=float, default=50.0, help="Overlap %% (default: 50)")
    p.add_argument("--var", type=str, default=None, help="Variable name in MAT file")
    p.add_argument("--outdir", type=str, default=".", help="Output directory")
    p.add_argument("--fmt", choices=["npz", "mat"], default="npz", help="Output format")

    p.add_argument("--bin", action="store_true", help="Apply octave-band binning")
    p.add_argument("--bpo", type=int, default=3, help="Bins per octave (default: 3)")

    p.add_argument("--correlation", action="store_true", help="Compute auto/cross-correlation")
    p.add_argument("--plot", action="store_true", help="Generate spectral plots")

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    # Expand globs on Windows
    files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        files.extend(expanded if expanded else [pattern])

    print(f"CSM Batch Processor v1.0")
    print(f"  Files:    {len(files)}")
    print(f"  fs:       {args.fs} Hz")
    print(f"  Nrec:     {args.nrec}")
    print(f"  Overlap:  {args.overlap}%")
    print(f"  Binning:  {'Yes (' + str(args.bpo) + ' bpo)' if args.bin else 'No'}")
    print(f"  Correlat: {'Yes' if args.correlation else 'No'}")
    print()

    for i, fpath in enumerate(files, 1):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        print(f"[{i}/{len(files)}] {os.path.basename(fpath)} ... ", end="", flush=True)

        t0 = time.time()
        try:
            data = load_data(fpath, variable_name=args.var)
            n_samples, n_channels = data.shape
            print(f"({n_samples} samples × {n_channels} ch) ", end="", flush=True)

            # --- Narrowband CSM ---
            spectra, freq = csm_calculator(data, args.fs, args.nrec, args.overlap)

            extras = {}

            # --- Octave-band binning ---
            if args.bin:
                df = args.fs / args.nrec
                freq_binned, spectra_binned = bin_csm(df, spectra, args.bpo)
                extras["freq_binned"] = freq_binned
                extras["spectra_binned"] = spectra_binned

            # --- Correlation ---
            if args.correlation:
                tau, corr_mat = compute_correlation(data, args.fs)
                extras["tau"] = tau
                extras["corr_matrix"] = corr_mat

            # --- Save ---
            outpath = os.path.join(args.outdir, f"{basename}_CSM")
            saved = save_results(outpath, spectra, freq, args.fs, args.nrec, args.overlap, args.fmt, **extras)

            # --- Plots ---
            if args.plot:
                fig = plot_autospectra(freq, spectra, title=f"Auto-Spectra — {basename}")
                fig.savefig(os.path.join(args.outdir, f"{basename}_autospectra.png"), dpi=150)
                plt.close(fig)

                if n_channels >= 2:
                    fig = plot_coherence(freq, spectra, 0, 1, title=f"Coherence Ch1–Ch2 — {basename}")
                    fig.savefig(os.path.join(args.outdir, f"{basename}_coherence.png"), dpi=150)
                    plt.close(fig)

                if args.correlation:
                    fig = plot_correlation(tau, corr_mat, title=f"Auto-Correlation — {basename}")
                    fig.savefig(os.path.join(args.outdir, f"{basename}_correlation.png"), dpi=150)
                    plt.close(fig)

            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s) → {saved}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\nDone!")


# Need this import at module level for --plot
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

if __name__ == "__main__":
    main()
