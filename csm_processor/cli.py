#!/usr/bin/env python3
"""Command-line interface for CSM Batch Processor."""

import argparse
import glob
from pathlib import Path
import time

from ._version import __version__
from .correlation import compute_correlation
from .csm_calculator import csm_calculator
from .io_utils import load_data, save_results
from .log_binning import bin_csm


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="csm-processor",
        description="Cross-Spectral Matrix Batch Processor",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("files", nargs="+", help="Input file(s): CSV, MAT, or TDMS")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--nrec", type=int, required=True, help="Record length in samples")
    parser.add_argument(
        "--overlap",
        type=float,
        default=50.0,
        help="Block overlap in percent (default: 50)",
    )
    parser.add_argument("--var", type=str, default=None, help="Variable name for MAT input")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("--fmt", choices=["npz", "mat"], default="npz", help="Output format")
    parser.add_argument("--bin", action="store_true", help="Apply octave-band binning")
    parser.add_argument("--bpo", type=int, default=3, help="Bins per octave (default: 3)")
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Compute auto/cross-correlation",
    )
    parser.add_argument("--plot", action="store_true", help="Save plots as PNG files")
    return parser.parse_args(argv)


def _validate_args(args):
    if args.fs <= 0:
        raise SystemExit("--fs must be positive.")
    if args.nrec <= 1:
        raise SystemExit("--nrec must be greater than 1.")
    if not 0 <= args.overlap < 100:
        raise SystemExit("--overlap must be in the range [0, 100).")
    if args.bin and args.bpo <= 0:
        raise SystemExit("--bpo must be positive when --bin is used.")


def _expand_input_files(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches or [pattern])

    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Input file(s) not found: {', '.join(missing)}")

    return files


def _print_run_header(args, files):
    print(f"CSM Batch Processor v{__version__}")
    print(f"  Files:        {len(files)}")
    print(f"  fs:           {args.fs} Hz")
    print(f"  Nrec:         {args.nrec}")
    print(f"  Overlap:      {args.overlap}%")
    print(f"  Binning:      {'Yes (' + str(args.bpo) + ' bpo)' if args.bin else 'No'}")
    print(f"  Correlation:  {'Yes' if args.correlation else 'No'}")
    print()


def _load_plotting_tools():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Matplotlib is required for --plot.") from exc

    from .plotting import plot_autospectra, plot_coherence, plot_correlation

    return plt, plot_autospectra, plot_coherence, plot_correlation


def _save_plots(plotting, outdir, basename, freq, spectra, n_channels, tau=None, corr_matrix=None):
    plt, plot_autospectra, plot_coherence, plot_correlation = plotting
    outdir = Path(outdir)

    fig = plot_autospectra(freq, spectra, title=f"Auto-Spectra - {basename}")
    fig.savefig(outdir / f"{basename}_autospectra.png", dpi=150)
    plt.close(fig)

    if n_channels >= 2:
        fig = plot_coherence(freq, spectra, 0, 1, title=f"Coherence Ch1-Ch2 - {basename}")
        fig.savefig(outdir / f"{basename}_coherence.png", dpi=150)
        plt.close(fig)

    if tau is not None and corr_matrix is not None:
        fig = plot_correlation(tau, corr_matrix, title=f"Auto-Correlation - {basename}")
        fig.savefig(outdir / f"{basename}_correlation.png", dpi=150)
        plt.close(fig)


def _process_file(fpath, args, plotting=None):
    data = load_data(fpath, variable_name=args.var)
    n_samples, n_channels = data.shape

    spectra, freq = csm_calculator(data, args.fs, args.nrec, args.overlap)
    extras = {}

    tau = None
    corr_matrix = None

    if args.bin:
        df = args.fs / args.nrec
        freq_binned, spectra_binned = bin_csm(df, spectra, bins_per_octave=args.bpo)
        extras["freq_binned"] = freq_binned
        extras["spectra_binned"] = spectra_binned

    if args.correlation:
        tau, corr_matrix = compute_correlation(data, args.fs)
        extras["tau"] = tau
        extras["corr_matrix"] = corr_matrix

    basename = Path(fpath).stem
    outpath = Path(args.outdir) / f"{basename}_CSM"

    saved = save_results(
        str(outpath),
        spectra,
        freq,
        args.fs,
        args.nrec,
        args.overlap,
        args.fmt,
        **extras,
    )

    if plotting is not None:
        _save_plots(
            plotting,
            args.outdir,
            basename,
            freq,
            spectra,
            n_channels,
            tau=tau,
            corr_matrix=corr_matrix,
        )

    return n_samples, n_channels, saved


def main(argv=None):
    args = _parse_args(argv)
    _validate_args(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = _expand_input_files(args.files)
    plotting = _load_plotting_tools() if args.plot else None

    _print_run_header(args, files)

    for index, fpath in enumerate(files, start=1):
        filename = Path(fpath).name
        print(f"[{index}/{len(files)}] {filename} ... ", end="", flush=True)
        start = time.time()

        try:
            n_samples, n_channels, saved = _process_file(fpath, args, plotting=plotting)
            elapsed = time.time() - start
            print(f"({n_samples} samples x {n_channels} ch) ✓ ({elapsed:.1f}s) -> {saved}")
        except Exception as exc:
            print(f"✗ Error: {exc}")

    print("\nDone!")


if __name__ == "__main__":
    main()