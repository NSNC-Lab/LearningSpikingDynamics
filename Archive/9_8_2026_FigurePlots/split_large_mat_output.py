"""Split an oversized MAT-v5 output array into MATLAB-loadable variables.

SciPy writes MAT-v5/v7 files, where each individual variable must remain
smaller than 2^31 bytes. This script preserves all other variables and splits
the first dimension of ``output`` into independently compressed variables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat, whosmat


DEFAULT_MAX_PART_BYTES = 1_000_000_000


def split_file(source: Path, destination: Path, max_part_bytes: int) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    print(f"Loading {source} ...", flush=True)
    contents = loadmat(source)
    if "output" not in contents:
        raise KeyError("The source MAT file does not contain an 'output' variable.")

    output = contents.pop("output")
    if output.ndim < 1 or output.shape[0] == 0:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    bytes_per_batch = output[0:1].nbytes
    batches_per_part = max(1, max_part_bytes // bytes_per_batch)
    if batches_per_part * bytes_per_batch >= 2**31:
        batches_per_part = max(1, (2**31 - 1) // bytes_per_batch)

    part_names: list[str] = []
    for part_number, start in enumerate(
        range(0, output.shape[0], batches_per_part), start=1
    ):
        stop = min(start + batches_per_part, output.shape[0])
        name = f"output_part_{part_number:03d}"
        part = output[start:stop]
        if part.nbytes >= 2**31:
            raise RuntimeError(f"{name} is still too large: {part.nbytes} bytes")
        contents[name] = part
        part_names.append(name)
        print(
            f"  {name}: batches {start}:{stop}, "
            f"shape={part.shape}, {part.nbytes / 2**30:.3f} GiB",
            flush=True,
        )

    contents["output_original_shape"] = np.asarray(output.shape, dtype=np.int64)
    contents["output_parts_count"] = np.asarray(len(part_names), dtype=np.int32)

    temporary = destination.with_name(destination.name + ".tmp")
    try:
        print(f"Writing {temporary} ...", flush=True)
        savemat(
            temporary,
            contents,
            do_compression=True,
            long_field_names=True,
            appendmat=False,
        )
        saved_names = {name for name, _, _ in whosmat(temporary)}
        missing = set(part_names) - saved_names
        if missing or "output" in saved_names:
            raise RuntimeError(
                f"Conversion verification failed; missing={sorted(missing)}, "
                f"unsplit_output_present={'output' in saved_names}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Converted file written to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original MAT file")
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help="Output path (default: SOURCE_split.mat)",
    )
    parser.add_argument(
        "--max-part-bytes",
        type=int,
        default=DEFAULT_MAX_PART_BYTES,
        help="Maximum uncompressed bytes per output part",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = (
        args.destination.expanduser().resolve()
        if args.destination
        else source.with_name(f"{source.stem}_split.mat")
    )
    split_file(source, destination, args.max_part_bytes)


if __name__ == "__main__":
    main()
