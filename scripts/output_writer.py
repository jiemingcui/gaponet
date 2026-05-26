#!/usr/bin/env python3
# Copyright (c) 2022-2025, GapONet Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
GapONet Output Writer
======================
Utility module for writing standardized machine-readable output artifacts.
Import this in deploy.py, play.py, or any evaluation script to ensure
consistent artifact contract across all modes.

Usage example (in deploy.py):

    from scripts.output_writer import GapONetOutputWriter

    writer = GapONetOutputWriter(output_dir="./runs/exp1")
    writer.write_eval_metrics({
        "large_gap_ratio": 0.12,
        "gap_iqr":         0.05,
        "gap_range":       0.22,
        "mpjae_deg":       3.4,
        "eef_error":       0.008,
        "per_payload":     {...}
    })
    writer.finalize(status="success")
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


class GapONetOutputWriter:
    """
    Writes standardized output artifacts to output_dir.

    Artifact contract:
        run_manifest.json       Always written (finalize() call)
        eval_metrics.json       write_eval_metrics()
        training_metrics.json   write_training_metrics()
        model_manifest.json     write_model_manifest()
        stdout.log              Handled externally (subprocess tee)
        stderr.log              Handled externally (subprocess tee)
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, output_dir, mode="unknown", run_id=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode    = mode
        self.run_id  = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start  = time.time()
        self._status = "running"

    # ─── Public write methods ───────────────────────────────────────

    def write_eval_metrics(self, metrics: dict):
        """
        Write eval_metrics.json.

        Recommended keys (all optional but encouraged):
          large_gap_ratio  float  Ratio of joint errors >= threshold
          gap_iqr          float  Interquartile range of joint errors
          gap_range        float  Range (max-min) of joint errors
          mpjae_deg        float  Mean Per-Joint Angle Error in degrees
          eef_error        float  End-effector position error (m)
          per_payload      dict   Per-payload-mass breakdown
          per_joint        dict   Per-joint breakdown
          threshold_rad    float  Threshold used for large_gap_ratio
          num_frames       int    Number of frames evaluated
          num_joints       int    Number of joints evaluated
        """
        out = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp":      datetime.now().isoformat(),
            "mode":           self.mode,
            "run_id":         self.run_id,
            **metrics,
        }
        self._write("eval_metrics.json", out)
        return out

    def write_training_metrics(self, metrics: dict):
        """
        Write training_metrics.json.

        Recommended keys:
          final_iteration      int
          mean_reward          float
          latest_checkpoint    str
          experiment_name      str
          run_name             str
          task                 str
          num_envs             int
          elapsed_seconds      float
        """
        out = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp":      datetime.now().isoformat(),
            "run_id":         self.run_id,
            **metrics,
        }
        self._write("training_metrics.json", out)
        return out

    def write_model_manifest(self, metrics: dict):
        """
        Write model_manifest.json.

        Recommended keys:
          source_checkpoint  str
          exported_model     str
          model_format       str   e.g. "torchscript_jit"
          task               str
          device             str
          input_shapes       dict  {branch: [...], trunk: [...]}
          output_dim         int
        """
        out = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp":      datetime.now().isoformat(),
            "run_id":         self.run_id,
            **metrics,
        }
        self._write("model_manifest.json", out)
        return out

    def write_per_joint_csv(self, rows: list, filename="per_joint_gap.csv"):
        """
        Write per-joint gap CSV. rows is a list of dicts with keys:
          joint_name, mean_gap_rad, std_gap_rad, large_gap_ratio, mpjae_deg
        """
        import csv
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        out_path = self.output_dir / filename
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[GapONet] Wrote {out_path}", flush=True)

    def finalize(self, status="success", extra=None):
        """
        Write run_manifest.json. Call this at the end of every run.
        """
        elapsed = time.time() - self._start
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "gaponet_version": "0.1.0",
            "timestamp":       datetime.now().isoformat(),
            "run_id":          self.run_id,
            "mode":            self.mode,
            "status":          status,
            "elapsed_seconds": round(elapsed, 2),
            "output_dir":      str(self.output_dir),
            "artifacts":       self._list_artifacts(),
        }
        if extra:
            manifest.update(extra)
        self._write("run_manifest.json", manifest)
        print(f"[GapONet] Run manifest written → {self.output_dir / 'run_manifest.json'}",
              flush=True)
        return manifest

    # ─── Internal ───────────────────────────────────────────────────

    def _write(self, filename, data):
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"[GapONet] Wrote {path}", flush=True)

    def _list_artifacts(self):
        return [f.name for f in sorted(self.output_dir.iterdir()) if f.is_file()]


# ─────────────────────────────────────────────
# Standalone helper for quick metric logging
# ─────────────────────────────────────────────

def log_eval_metrics(output_dir, metrics: dict, mode="deploy"):
    """
    One-liner helper. Creates writer, writes eval_metrics.json, finalizes.

    Example:
        from scripts.output_writer import log_eval_metrics
        log_eval_metrics("./runs/exp1", {
            "large_gap_ratio": 0.12,
            "mpjae_deg": 3.4,
        })
    """
    w = GapONetOutputWriter(output_dir, mode=mode)
    w.write_eval_metrics(metrics)
    return w.finalize(status="success")
