#!/usr/bin/env python
"""Extract rendered figures from an executed notebook into ``assets/`` for the README.

Kaggle renders the gallery / metric / panel figures inline; this pulls those image
outputs out of a downloaded ``*.ipynb`` so they can be embedded in the README without
committing the (heavy) executed notebook itself.

Usage:
    python scripts/extract_notebook_figures.py path/to/executed.ipynb --prefix method_comparison
    # -> assets/method_comparison_1.png, assets/method_comparison_2.png, ...
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", help="executed .ipynb (downloaded from Kaggle, with outputs)")
    p.add_argument("--prefix", required=True, help="output filename stem under assets/")
    p.add_argument("--out", default=str(REPO / "assets"), help="output directory (default: assets/)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(args.notebook, as_version=4)

    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if not png:
                continue
            n += 1
            dest = out_dir / f"{args.prefix}_{n}.png"
            dest.write_bytes(base64.b64decode(png))
            print("wrote", dest)

    if not n:
        print("no image/png outputs found -- is this the executed notebook (with figures)?")
    else:
        print(f"\n{n} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()
