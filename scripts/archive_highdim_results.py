#!/usr/bin/env python3
"""
Compresses and archives all highdim EI benchmark results, manifest, markdown reports,
CSV tables, and raw telemetry JSONs into a single versioned tarball.
"""

import os
import sys
import tarfile
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "latest"

def archive_highdim_results(
    source_dir: str = "results/epistemic_ei_highdim",
    output_tarball: str = None
) -> str:
    if output_tarball is None:
        commit = get_git_commit()
        output_tarball = f"results/epistemic_ei_highdim_archive_{commit}.tar.gz"

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    os.makedirs(os.path.dirname(output_tarball) if os.path.dirname(output_tarball) else ".", exist_ok=True)

    print(f"Archiving '{source_dir}' into '{output_tarball}'...")
    with tarfile.open(output_tarball, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    print(f"Archive successfully created: {output_tarball} ({os.path.getsize(output_tarball) / (1024*1024):.2f} MB)")
    return output_tarball

def main():
    parser = argparse.ArgumentParser(description="Archive High-Dim Benchmark Results")
    parser.add_argument("--source_dir", type=str, default="results/epistemic_ei_highdim")
    parser.add_argument("--output_tarball", type=str, default=None)
    args = parser.parse_args()

    archive_highdim_results(source_dir=args.source_dir, output_tarball=args.output_tarball)

if __name__ == "__main__":
    main()
