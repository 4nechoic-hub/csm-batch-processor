"""
I/O Utilities
==============
Load time-series data from CSV, MAT, or TDMS files, and save results to
MAT or NPZ format.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np


def load_data(
    filepath: str,
    variable_name: Optional[str] = None,
) -> np.ndarray:
    """
    Load multi-channel time-series data from a file.

    Supported formats
    -----------------
    - **.csv / .tsv / .txt** — delimited text (auto-detected separator).
      Numeric columns are treated as channels.  A single header row is
      automatically skipped if detection indicates one.
    - **.mat** — MATLAB v5/v7/v7.3 files.  If *variable_name* is given that
      variable is loaded; otherwise the first 2-D numeric array found is used.
    - **.tdms** — NI TDMS files (requires the ``npTDMS`` package).

    Returns
    -------
    data : np.ndarray, shape (N, M)
        Time-series matrix (N samples × M channels).
    """

    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".csv", ".tsv", ".txt", ".dat"):
        return _load_delimited(filepath)
    elif ext == ".mat":
        return _load_mat(filepath, variable_name)
    elif ext == ".tdms":
        return _load_tdms(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# --------------------------------------------------------------------------- #
#  Private loaders
# --------------------------------------------------------------------------- #


def _load_delimited(filepath: Path) -> np.ndarray:
    """Load a delimited text file, auto-detecting separator and header."""
    import csv

    with open(filepath, "r") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample)
    has_header = csv.Sniffer().has_header(sample)

    data = np.genfromtxt(
        filepath,
        delimiter=dialect.delimiter,
        skip_header=1 if has_header else 0,
    )

    # Drop any columns that are all-NaN (e.g. trailing delimiters)
    valid_cols = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid_cols]

    if data.ndim == 1:
        data = data[:, np.newaxis]

    return data


def _load_mat(filepath: Path, variable_name: Optional[str] = None) -> np.ndarray:
    """Load a MATLAB .mat file (v5–v7.3)."""
    try:
        import scipy.io as sio

        mat = sio.loadmat(str(filepath))
        return _extract_mat_var(mat, variable_name)
    except NotImplementedError:
        # v7.3 (HDF5-based)
        import h5py  # type: ignore

        with h5py.File(str(filepath), "r") as f:
            if variable_name and variable_name in f:
                return np.array(f[variable_name]).T  # HDF5 stores col-major
            # Auto-detect first numeric dataset
            for key in f:
                ds = f[key]
                if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.number):
                    return np.array(ds).T
        raise ValueError("No suitable numeric variable found in MAT file.")


def _extract_mat_var(mat: dict, variable_name: Optional[str]) -> np.ndarray:
    """Pick the right variable from a loaded mat-dict."""
    # Filter out MATLAB metadata keys
    keys = [k for k in mat if not k.startswith("__")]

    if variable_name:
        if variable_name not in mat:
            raise KeyError(
                f"Variable '{variable_name}' not found. Available: {keys}"
            )
        return np.asarray(mat[variable_name], dtype=float)

    # Auto-detect: pick the largest 2-D numeric array
    best_key, best_size = None, 0
    for k in keys:
        arr = mat[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
            if arr.size > best_size:
                best_key, best_size = k, arr.size

    if best_key is None:
        raise ValueError(f"No numeric arrays found. Keys: {keys}")

    data = np.asarray(mat[best_key], dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data


def _load_tdms(filepath: Path) -> np.ndarray:
    """Load an NI TDMS file using npTDMS."""
    try:
        from nptdms import TdmsFile  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'nptdms' package is required to read TDMS files.\n"
            "Install it with: pip install npTDMS"
        )

    tdms = TdmsFile.read(str(filepath))
    channels = []
    for group in tdms.groups():
        for channel in group.channels():
            channels.append(channel.data)

    if not channels:
        raise ValueError("No data channels found in TDMS file.")

    # Stack as columns — truncate to shortest channel if lengths differ
    min_len = min(len(c) for c in channels)
    data = np.column_stack([c[:min_len] for c in channels])
    return data


# --------------------------------------------------------------------------- #
#  Save helpers
# --------------------------------------------------------------------------- #


def save_results(
    filepath: str,
    spectra: np.ndarray,
    freq: np.ndarray,
    fs: float,
    n_rec: int,
    overlap: float,
    fmt: str = "npz",
    **extra_arrays,
) -> str:
    """
    Save CSM results to disk.

    Parameters
    ----------
    filepath : str
        Output path (extension is overridden by *fmt*).
    spectra, freq : np.ndarray
        CSM and frequency vector.
    fs, n_rec, overlap : float / int
        Processing parameters.
    fmt : str
        ``'npz'`` (default) or ``'mat'``.
    **extra_arrays
        Any additional arrays to include (e.g. binned spectra, correlation).

    Returns
    -------
    saved_path : str
        Actual path written.
    """
    base = os.path.splitext(filepath)[0]

    payload = dict(
        spectra=spectra,
        freq=freq,
        fs=np.float64(fs),
        n_rec=np.int64(n_rec),
        overlap=np.float64(overlap),
        **extra_arrays,
    )

    if fmt == "mat":
        import scipy.io as sio

        out = base + ".mat"
        sio.savemat(out, payload, do_compression=True)
    else:
        out = base + ".npz"
        np.savez_compressed(out, **payload)

    return out
