"""Helpers for listing and grouping .npz files by their saved metadata.

The C-RED ONE GUI saves files through cred_controller.py with
``np.savez(..., image=image_array, timestamp=..., **metadata)``.
Each .npz therefore contains the image array plus metadata values such as
``fps``, ``mode``, ``gain``, etc.

Example
-------
>>> from analysis.npz_metadata_utils import group_npz_files_by_metadata
>>> groups = group_npz_files_by_metadata("/path/to/data", "fps")
>>> list_a = groups[100.0]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def read_npz_metadata(npz_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Read the metadata stored in an NPZ file.

    The image array is skipped, and other saved values are normalized to
    native Python scalars or lists when possible.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        metadata: Dict[str, Any] = {}
        for key in data.files:
            if key == "image":
                continue
            value = data[key]
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    metadata[key] = value.item()
                else:
                    metadata[key] = value.tolist()
            else:
                metadata[key] = value
        return metadata


def list_npz_files(
    path: str | os.PathLike[str],
    pattern: str = "*.npz",
    recursive: bool = False,
) -> List[str]:
    """Return a sorted list of .npz files under a folder."""
    root = Path(path)
    if recursive:
        matches = root.rglob(pattern)
    else:
        matches = root.glob(pattern)

    return sorted(str(item) for item in matches if item.is_file())


def group_npz_files_by_metadata(
    path: str | os.PathLike[str],
    metadata_key: str,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> Dict[Any, List[str]]:
    """Group .npz files by the value of one metadata field.

    Parameters
    ----------
    path
        Folder containing the .npz files.
    metadata_key
        Metadata key to group by, for example ``fps`` or ``mode``.
    pattern
        File pattern to search for.
    recursive
        Whether to search subfolders as well.

    Returns
    -------
    dict
        Mapping from metadata value to a list of file names.
    """
    groups: Dict[Any, List[str]] = {}

    for npz_file in list_npz_files(path, pattern=pattern, recursive=recursive):
        metadata = read_npz_metadata(npz_file)
        value = metadata.get(metadata_key, "<missing>")
        groups.setdefault(value, []).append(os.path.basename(npz_file))

    return groups


def get_npz_files_by_metadata(
    path: str | os.PathLike[str],
    metadata_key: str,
    metadata_value: Any,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> List[str]:
    """Return the list of .npz files that share a metadata value."""
    groups = group_npz_files_by_metadata(
        path,
        metadata_key,
        pattern=pattern,
        recursive=recursive,
    )
    return groups.get(metadata_value, [])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Group .npz files by saved metadata")
    parser.add_argument("path", help="Folder containing .npz files")
    parser.add_argument(
        "metadata_key",
        nargs="?",
        default="fps",
        help="Metadata field to group by (default: fps)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders as well",
    )
    args = parser.parse_args()

    groups = group_npz_files_by_metadata(
        args.path,
        args.metadata_key,
        recursive=args.recursive,
    )

    for value, files in sorted(groups.items(), key=lambda item: str(item[0])):
        print(f"{value}: {len(files)} file(s)")
        for filename in files:
            print(f"  - {filename}")
