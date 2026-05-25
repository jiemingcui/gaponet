#!/usr/bin/env python3
# Copyright (c) 2022-2025, GapONet Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
deploy.py — Output Writer Integration Patch
=============================================
This file shows EXACTLY what to add to your existing deploy.py to make it
emit standardized machine-readable artifacts (eval_metrics.json, run_manifest.json).

HOW TO APPLY:
  1. Open your existing scripts/rsl_rl/deploy.py
  2. Copy the "ADD THIS" blocks below into the appropriate locations
  3. That's it — no other changes needed

The patch is minimal: it only adds output writing at the end of your
existing evaluation loop without modifying any logic.
"""

# ══════════════════════════════════════════════════════════
# ❶  ADD THIS near the top of deploy.py (after your imports)
# ══════════════════════════════════════════════════════════

import sys
import os
from pathlib import Path

# Make scripts/ importable regardless of cwd
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # → repo_root/scripts
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from output_writer import GapONetOutputWriter
    _HAS_OUTPUT_WRITER = True
except ImportError:
    _HAS_OUTPUT_WRITER = False


# ══════════════════════════════════════════════════════════
# ❷  ADD THIS to your argument parser in deploy.py
# ══════════════════════════════════════════════════════════

def _add_output_args(parser):
    """Call this inside your existing argparse setup."""
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write eval_metrics.json and run_manifest.json. "
             "If not set, writes to ./gaponet_output/<timestamp>/")
    return parser


# ══════════════════════════════════════════════════════════
# ❸  ADD THIS at the END of your main() / evaluation loop
#    after you have computed all metrics
# ══════════════════════════════════════════════════════════

def write_deploy_outputs(args, results: dict):
    """
    Call at the end of deploy.py's evaluation loop.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args from your existing argparse (must include output_dir
        after you add it with _add_output_args above).
    results : dict
        Flat dict of computed metrics. Recommended keys:
          large_gap_ratio   float
          gap_iqr           float
          gap_range         float
          mpjae_deg         float
          eef_error         float
          per_payload       dict   {mass_kg: {large_gap_ratio, mpjae_deg, ...}}
          per_joint         list   [{joint_name, mean_gap_rad, ...}, ...]
          threshold_rad     float
          num_frames        int
          num_joints        int
    """
    if not _HAS_OUTPUT_WRITER:
        print("[GapONet] output_writer not found — skipping artifact writing.",
              flush=True)
        return

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("gaponet_output") / f"deploy_{ts}"

    writer = GapONetOutputWriter(output_dir=output_dir, mode="deploy")

    # Write eval metrics
    writer.write_eval_metrics({
        "large_gap_ratio": results.get("large_gap_ratio"),
        "gap_iqr":         results.get("gap_iqr"),
        "gap_range":       results.get("gap_range"),
        "mpjae_deg":       results.get("mpjae_deg"),
        "eef_error":       results.get("eef_error"),
        "threshold_rad":   results.get("threshold_rad", 0.5),
        "num_frames":      results.get("num_frames"),
        "num_joints":      results.get("num_joints"),
        "per_payload":     results.get("per_payload"),
        "test_data":       str(getattr(args, "test_data", "")),
        "model":           str(getattr(args, "model", "")),
    })

    # Write per-joint CSV if available
    per_joint = results.get("per_joint")
    if per_joint and isinstance(per_joint, list):
        writer.write_per_joint_csv(per_joint)

    writer.finalize(status="success", extra={
        "test_data": str(getattr(args, "test_data", "")),
        "model":     str(getattr(args, "model", "")),
    })

    print(f"\n[GapONet] ✓ Artifacts written to: {output_dir}", flush=True)


# ══════════════════════════════════════════════════════════
# COMPLETE USAGE EXAMPLE
# (shows what your deploy.py main() should look like after patching)
# ══════════════════════════════════════════════════════════

def _example_patched_main():
    """
    EXAMPLE ONLY — not meant to be run directly.
    Shows the minimal additions to your existing deploy.py main().
    """
    import argparse

    parser = argparse.ArgumentParser()
    # --- your existing args ---
    parser.add_argument("--model",     type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    # --- ADD THIS LINE ---
    _add_output_args(parser)

    args = parser.parse_args()

    # ... your existing inference and metric computation ...
    # (unchanged)

    # After computing metrics, build the results dict:
    results = {
        # --- populate from your existing variables ---
        "large_gap_ratio": 0.0,   # replace with your computed value
        "gap_iqr":         0.0,
        "gap_range":       0.0,
        "mpjae_deg":       0.0,
        "eef_error":       0.0,
        "per_payload":     {},    # your existing per-payload dict
        "per_joint":       [],    # list of per-joint dicts (optional)
        "num_frames":      0,
        "num_joints":      0,
    }

    # --- ADD THIS CALL at the end ---
    write_deploy_outputs(args, results)
