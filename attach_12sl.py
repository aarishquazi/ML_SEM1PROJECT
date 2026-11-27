#!/usr/bin/env python3
"""Attach PTB-XL+ 12SL `statements_ext` labels to the existing ECG image manifest.

The join is performed by ecg_id (never by row position), so input CSV files may
be in completely different orders. The output is a compact training index:
    ecg_id, image_path, label_<statement_code> ...

Every statement code present in statements_ext is marked 1, regardless of its
certainty value. No image conversion is performed and no images are copied.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default paths relative to the project root.
DEFAULT_MANIFEST = Path(
    "data/processed/12sl_median_images_final/manifest_unlabelled.csv"
)
DEFAULT_STATEMENTS = Path(
    "data/ptb-xl-plus/labels/12sl_statements.csv"
)
DEFAULT_OUTPUT = Path(
    "data/processed/12sl_median_images_final/training_index.csv"
)


def parse_statement_ext(value: Any, row_id: int) -> list[str]:
    """Parse a CSV cell such as [('NSR', 100.0), ('NML', 100.0)]."""
    if pd.isna(value):
        raise ValueError(f"ecg_id={row_id}: statements_ext is empty")

    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"ecg_id={row_id}: cannot parse statements_ext: {value!r}"
        ) from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"ecg_id={row_id}: statements_ext must be a list of (code, certainty) pairs"
        )

    codes: list[str] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                f"ecg_id={row_id}: malformed statement entry: {item!r}"
            )

        code = str(item[0]).strip()
        if not code:
            raise ValueError(f"ecg_id={row_id}: empty statement code")

        # Validate certainty exists and is numeric, but do not treat it as a
        # probability or filter labels by it in this binary-target experiment.
        try:
            certainty = float(item[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ecg_id={row_id}: invalid certainty for {code}: {item[1]!r}"
            ) from exc
        if not np.isfinite(certainty):
            raise ValueError(
                f"ecg_id={row_id}: non-finite certainty for {code}: {item[1]!r}"
            )

        codes.append(code)

    if not codes:
        raise ValueError(f"ecg_id={row_id}: no statement codes found")

    # De-duplicate within a record while preserving first occurrence.
    return list(dict.fromkeys(codes))


def read_ecg_ids(frame: pd.DataFrame, filename: str) -> None:
    """Fail early on missing, non-integer, or duplicate ECG identifiers."""
    if frame["ecg_id"].isna().any():
        raise ValueError(f"{filename}: ecg_id contains missing values")

    numeric_ids = pd.to_numeric(frame["ecg_id"], errors="coerce")
    if numeric_ids.isna().any() or not np.all(numeric_ids == np.floor(numeric_ids)):
        raise ValueError(f"{filename}: ecg_id must contain integer values")

    frame["ecg_id"] = numeric_ids.astype("int64")

    duplicates = frame.loc[frame["ecg_id"].duplicated(), "ecg_id"].head(10).tolist()
    if duplicates:
        raise ValueError(
            f"{filename}: duplicate ecg_id values detected (examples: {duplicates})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a minimal multi-label training index for existing 12SL ECG images."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--statements", type=Path, default=DEFAULT_STATEMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip checking that each image_path exists relative to the manifest folder.",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.statements.is_file():
        print(f"ERROR: 12SL statements CSV not found: {args.statements}", file=sys.stderr)
        return 2

    # Only load columns needed for training: ECG ID and image path.
    manifest = pd.read_csv(
        args.manifest,
        usecols=["ecg_id", "image_path"],
        dtype={"image_path": "string"},
    )
    read_ecg_ids(manifest, args.manifest.name)

    if manifest["image_path"].isna().any() or (manifest["image_path"].str.strip() == "").any():
        raise ValueError("Manifest contains empty image_path values")

    # Only load ECG ID and the selected target source.
    statements = pd.read_csv(
        args.statements,
        usecols=["ecg_id", "statements_ext"],
    )
    read_ecg_ids(statements, args.statements.name)

    # Parse statements by ECG ID. This does NOT rely on either CSV being sorted.
    statements["_codes"] = [
        parse_statement_ext(value, int(ecg_id))
        for ecg_id, value in zip(statements["ecg_id"], statements["statements_ext"])
    ]
    statements = statements[["ecg_id", "_codes"]]

    # Left-join from the image manifest so every existing image is retained.
    # validate='one_to_one' protects against accidental duplicate-ID joins.
    joined = manifest.merge(
        statements,
        on="ecg_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    missing_label_ids = joined.loc[joined["_merge"] != "both", "ecg_id"].tolist()
    if missing_label_ids:
        raise ValueError(
            f"{len(missing_label_ids)} images have no matching 12SL label row. "
            f"Example ecg_id values: {missing_label_ids[:10]}"
        )

    # Label vocabulary is created from the labels of the image-matched records.
    vocabulary = sorted({code for row_codes in joined["_codes"] for code in row_codes})
    if not vocabulary:
        raise ValueError("No target labels were found in statements_ext")

    # Construct compact multi-hot target columns (1 = code present; 0 = absent).
    code_to_col = {code: index for index, code in enumerate(vocabulary)}
    targets = np.zeros((len(joined), len(vocabulary)), dtype=np.uint8)
    for row_index, row_codes in enumerate(joined["_codes"]):
        for code in row_codes:
            targets[row_index, code_to_col[code]] = 1

    # Keep only information required to locate the image and train the model.
    result = pd.DataFrame({
        "ecg_id": joined["ecg_id"].to_numpy(dtype=np.int64),
        "image_path": joined["image_path"].astype(str).to_numpy(),
    })
    label_columns = [f"label_{code}" for code in vocabulary]
    target_frame = pd.DataFrame(targets, columns=label_columns)
    result = pd.concat([result, target_frame], axis=1)

    # Sort output for readability only. Correct matching has already happened
    # by ecg_id, so source-file row order is irrelevant.
    result = result.sort_values("ecg_id", kind="stable").reset_index(drop=True)

    # Check the saved image paths from the manifest's directory.
    if not args.skip_image_check:
        image_root = args.manifest.parent
        missing_images = []
        for image_path in result["image_path"]:
            path = Path(image_path)
            resolved = path if path.is_absolute() else image_root / path
            if not resolved.is_file():
                missing_images.append(image_path)
                if len(missing_images) >= 10:
                    break
        if missing_images:
            raise FileNotFoundError(
                "Some image files referenced by the manifest do not exist "
                f"(first examples: {missing_images}). Check the manifest/image folder "
                "or use --skip-image-check if the images are stored elsewhere."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    # Informational counts only; no extra training files are generated.
    label_records_without_images = len(
        set(statements["ecg_id"]) - set(manifest["ecg_id"])
    )
    positives = targets.sum(axis=0)
    labels_per_record = targets.sum(axis=1)

    print("\n12SL labelling completed")
    print(f"Image records: {len(manifest):,}")
    print(f"Matched labelled images: {len(result):,}")
    print(f"Unique target labels: {len(vocabulary):,}")
    print(f"Positive label assignments: {int(targets.sum()):,}")
    print(f"Average labels per ECG: {labels_per_record.mean():.3f}")
    print(f"Minimum labels per ECG: {int(labels_per_record.min())}")
    print(f"Maximum labels per ECG: {int(labels_per_record.max())}")
    print(f"12SL label rows without converted images: {label_records_without_images:,}")
    print(f"Output columns: {2 + len(vocabulary)} (ecg_id + image_path + labels)")
    print(f"Training index: {args.output}")
    print("\nNote: every code listed in statements_ext is treated as a positive binary target.")
    print("Certainty values are not used as probabilities or weights in this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
