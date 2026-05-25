#!/usr/bin/env python3
# Copyright (c) 2022-2025, GapONet Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
GapONet Unified Job Runner
===========================
A single deterministic CLI entry point for all GapONet operations.
Designed for both human use and agent/harness automation.

Modes:
  train       Train a policy with Isaac Sim
  eval        Evaluate a trained policy with Isaac Sim (play.py)
  export      Export a checkpoint to JIT/TorchScript format
  deploy      Run lightweight inference + evaluation (no Isaac Sim needed)

Usage:
  python scripts/run_gaponet_job.py --mode train   --config configs/train_default.json --output-dir ./runs/exp1
  python scripts/run_gaponet_job.py --mode eval    --config configs/eval_default.json  --output-dir ./runs/exp1
  python scripts/run_gaponet_job.py --mode export  --checkpoint ./model/model_17950.pt --output-dir ./runs/exp1
  python scripts/run_gaponet_job.py --mode deploy  --input-data ./data/test.npz        --output-dir ./runs/exp1

  # From a JSON input package (agent-friendly):
  python scripts/run_gaponet_job.py --input-package input_package.json --output-dir ./runs/exp1

Output contract (always written to --output-dir):
  run_manifest.json      — inputs, timing, status, exit code
  eval_metrics.json      — (eval/deploy) gap ratio, MPJAE, IQR, range, EEF error
  training_metrics.json  — (train) final iteration, reward, checkpoint path
  model_manifest.json    — (export) output model path, format, task
  stdout.log
  stderr.log
"""

import argparse
import json
import os
import sys
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = {
    "train":  REPO_ROOT / "scripts" / "rsl_rl" / "train.py",
    "eval":   REPO_ROOT / "scripts" / "rsl_rl" / "play.py",
    "export": REPO_ROOT / "scripts" / "rsl_rl" / "inference_jit.py",
    "deploy": REPO_ROOT / "scripts" / "rsl_rl" / "deploy.py",
}

DEFAULT_TASK = "Isaac-Humanoid-Operator-Delta-Action"

EXIT_CODES = {
    "success":              0,
    "invalid_args":         2,
    "missing_script":       3,
    "missing_checkpoint":   4,
    "missing_data":         5,
    "missing_config":       6,
    "subprocess_failed":    10,
    "unexpected_error":     99,
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _error(msg, code_key="unexpected_error"):
    _log(msg, "ERROR")
    return EXIT_CODES[code_key]


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_manifest(output_dir, mode, args_dict, status, exit_code,
                    elapsed_s, extra=None):
    manifest = {
        "gaponet_version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": round(elapsed_s, 2),
        "inputs": args_dict,
        "output_dir": str(output_dir),
    }
    if extra:
        manifest.update(extra)
    _write_json(Path(output_dir) / "run_manifest.json", manifest)


def _stream_subprocess(cmd, stdout_log, stderr_log, cwd=None):
    """Run a subprocess, tee output to files and terminal, return returncode."""
    _log(f"Running: {' '.join(str(c) for c in cmd)}")

    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or REPO_ROOT,
            text=True,
        )

        import threading

        def _tee(src, dst_file, dst_stream):
            for line in src:
                dst_file.write(line)
                dst_file.flush()
                dst_stream.write(line)
                dst_stream.flush()
            src.close()

        t_out = threading.Thread(target=_tee,
                                 args=(proc.stdout, fout, sys.stdout))
        t_err = threading.Thread(target=_tee,
                                 args=(proc.stderr, ferr, sys.stderr))
        t_out.start()
        t_err.start()
        t_out.join()
        t_err.join()
        proc.wait()

    return proc.returncode


# ─────────────────────────────────────────────
# Mode implementations
# ─────────────────────────────────────────────

def run_train(cfg, output_dir):
    script = SCRIPTS["train"]
    if not script.exists():
        return _error(
            f"train.py not found at {script}. "
            "Run ./sync_rsl_scripts.sh first.",
            "missing_script")

    task          = cfg.get("task", DEFAULT_TASK)
    num_envs      = cfg.get("num_envs", 4080)
    max_iterations= cfg.get("max_iterations", 100000)
    experiment    = cfg.get("experiment_name", "GapONet")
    run_name      = cfg.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    letter        = cfg.get("letter", "amass")
    device        = cfg.get("device", "cuda")
    headless      = cfg.get("headless", True)
    extra_args    = cfg.get("extra_args", [])

    cmd = [
        sys.executable, str(script),
        "--task", task,
        f"--num_envs={num_envs}",
        "--max_iterations", str(max_iterations),
        "--experiment_name", experiment,
        "--letter", letter,
        "--run_name", run_name,
        "--device", device,
        "env.mode=train",
    ]
    if headless:
        cmd.append("--headless")
    cmd.extend(extra_args)

    rc = _stream_subprocess(
        cmd,
        output_dir / "stdout.log",
        output_dir / "stderr.log")

    # Try to find the latest checkpoint produced
    ckpt_dir = REPO_ROOT / "logs" / experiment / run_name
    checkpoints = sorted(ckpt_dir.glob("model_*.pt")) if ckpt_dir.exists() else []
    latest_ckpt = str(checkpoints[-1]) if checkpoints else None

    metrics = {
        "task": task,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "experiment_name": experiment,
        "run_name": run_name,
        "latest_checkpoint": latest_ckpt,
        "subprocess_exit_code": rc,
    }
    _write_json(output_dir / "training_metrics.json", metrics)

    return 0 if rc == 0 else EXIT_CODES["subprocess_failed"]


def run_eval(cfg, output_dir):
    script = SCRIPTS["eval"]
    if not script.exists():
        return _error(
            f"play.py not found at {script}. "
            "Run ./sync_rsl_scripts.sh first.",
            "missing_script")

    task     = cfg.get("task", DEFAULT_TASK)
    model    = cfg.get("model") or cfg.get("checkpoint")
    num_envs = cfg.get("num_envs", 20)
    headless = cfg.get("headless", True)
    extra_args = cfg.get("extra_args", [])

    if not model:
        return _error("eval mode requires 'model' or 'checkpoint' in config",
                      "missing_checkpoint")
    if not Path(model).exists():
        return _error(f"Checkpoint not found: {model}", "missing_checkpoint")

    cmd = [
        sys.executable, str(script),
        "--task", task,
        "--model", str(model),
        "--num_envs", str(num_envs),
    ]
    if headless:
        cmd.append("--headless")
    cmd.extend(extra_args)

    rc = _stream_subprocess(
        cmd,
        output_dir / "stdout.log",
        output_dir / "stderr.log")

    # Parse any metrics play.py may have written (task-dependent)
    metrics = {
        "task": task,
        "model": str(model),
        "num_envs": num_envs,
        "subprocess_exit_code": rc,
    }
    _write_json(output_dir / "eval_metrics.json", metrics)

    return 0 if rc == 0 else EXIT_CODES["subprocess_failed"]


def run_export(cfg, output_dir):
    script = SCRIPTS["export"]
    if not script.exists():
        return _error(
            f"inference_jit.py not found at {script}.",
            "missing_script")

    checkpoint = cfg.get("checkpoint")
    task       = cfg.get("task", DEFAULT_TASK)
    device     = cfg.get("device", "cuda:0")
    num_envs   = cfg.get("num_envs", 20)
    out_model  = cfg.get("output_model") or str(output_dir / "policy.pt")

    if not checkpoint:
        return _error("export mode requires 'checkpoint' in config",
                      "missing_checkpoint")
    if not Path(checkpoint).exists():
        return _error(f"Checkpoint not found: {checkpoint}", "missing_checkpoint")

    cmd = [
        sys.executable, str(script),
        "--export",
        "--checkpoint", str(checkpoint),
        "--task", task,
        "--output", str(out_model),
        "--device", device,
        "--num_envs", str(num_envs),
    ]

    rc = _stream_subprocess(
        cmd,
        output_dir / "stdout.log",
        output_dir / "stderr.log")

    manifest = {
        "source_checkpoint": str(checkpoint),
        "exported_model":    str(out_model),
        "model_format":      "torchscript_jit",
        "task":              task,
        "device":            device,
        "subprocess_exit_code": rc,
    }
    _write_json(output_dir / "model_manifest.json", manifest)

    return 0 if rc == 0 else EXIT_CODES["subprocess_failed"]


def run_deploy(cfg, output_dir):
    """
    Lightweight inference + evaluation. No Isaac Sim required.
    Runs deploy.py and parses its output into eval_metrics.json.
    """
    script = SCRIPTS["deploy"]
    if not script.exists():
        return _error(
            f"deploy.py not found at {script}.",
            "missing_script")

    test_data  = cfg.get("test_data") or cfg.get("input_data")
    model      = cfg.get("model") or cfg.get("checkpoint")
    extra_args = cfg.get("extra_args", [])

    if not test_data:
        return _error("deploy mode requires 'test_data' or 'input_data' in config",
                      "missing_data")
    if not Path(test_data).exists():
        return _error(f"Test data not found: {test_data}", "missing_data")

    cmd = [sys.executable, str(script)]
    if model:
        if not Path(model).exists():
            return _error(f"Model not found: {model}", "missing_checkpoint")
        cmd += ["--model", str(model)]
    cmd += ["--test_data", str(test_data)]
    cmd.extend(extra_args)

    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"

    rc = _stream_subprocess(cmd, stdout_log, stderr_log)

    # Parse metrics from stdout (deploy.py prints tables)
    metrics = _parse_deploy_stdout(stdout_log)
    metrics["test_data"] = str(test_data)
    metrics["model"]     = str(model) if model else None
    metrics["subprocess_exit_code"] = rc
    _write_json(output_dir / "eval_metrics.json", metrics)

    return 0 if rc == 0 else EXIT_CODES["subprocess_failed"]


def _parse_deploy_stdout(stdout_log):
    """
    Best-effort parser for deploy.py stdout.
    Extracts numeric metric values by looking for known keywords.
    Returns dict; all values are optional.
    """
    metrics = {
        "large_gap_ratio":  None,
        "gap_iqr":          None,
        "gap_range":        None,
        "mpjae_deg":        None,
        "eef_error":        None,
    }
    try:
        text = Path(stdout_log).read_text(errors="replace")
        import re
        patterns = {
            "large_gap_ratio": r"[Ll]arge\s+[Gg]ap\s+[Rr]atio[:\s]+([0-9.]+)",
            "gap_iqr":         r"[Gg]ap\s+IQR[:\s]+([0-9.]+)",
            "gap_range":       r"[Gg]ap\s+[Rr]ange[:\s]+([0-9.]+)",
            "mpjae_deg":       r"MPJAE[:\s]+([0-9.]+)",
            "eef_error":       r"[Ee][Ee][Ff]\s+[Ee]rror[:\s]+([0-9.]+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                metrics[key] = float(m.group(1))
    except Exception:
        pass
    return metrics


# ─────────────────────────────────────────────
# Input package loader (agent-friendly path)
# ─────────────────────────────────────────────

def load_input_package(path):
    """
    Load a JSON input package. Expected schema:

    {
      "mode": "train|eval|export|deploy",
      "config": { ... mode-specific fields ... }
    }

    Or flat (all fields at top level, 'mode' required).
    """
    pkg = _load_json(path)
    mode = pkg.get("mode")
    cfg  = pkg.get("config", pkg)   # fall back to flat layout
    return mode, cfg


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GapONet unified job runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # ── Primary control ──
    parser.add_argument(
        "--mode", choices=["train", "eval", "export", "deploy"],
        help="Operation mode")
    parser.add_argument(
        "--input-package",
        help="Path to input_package.json (overrides all other flags)")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for all outputs (created if absent)")

    # ── Per-mode shorthands ──
    parser.add_argument("--config",
                        help="(train/eval) JSON config file path")
    parser.add_argument("--checkpoint",
                        help="(eval/export) Path to .pt checkpoint")
    parser.add_argument("--input-data",
                        help="(deploy) Path to .npz test data")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help=f"Isaac Lab task name (default: {DEFAULT_TASK})")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without display (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false")

    args = parser.parse_args()

    # ── Resolve mode + config ──
    if args.input_package:
        mode, cfg = load_input_package(args.input_package)
    else:
        mode = args.mode
        if not mode:
            parser.error("--mode is required (or use --input-package)")
        # Build cfg from CLI flags
        cfg = {"task": args.task, "device": args.device, "headless": args.headless}
        if args.config:
            if not Path(args.config).exists():
                sys.exit(_error(f"Config not found: {args.config}", "missing_config"))
            cfg.update(_load_json(args.config))
        if args.checkpoint:
            cfg["checkpoint"] = args.checkpoint
        if args.input_data:
            cfg["test_data"] = args.input_data
        if args.num_envs is not None:
            cfg["num_envs"] = args.num_envs

    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    _log(f"Mode: {mode}")
    _log(f"Output dir: {output_dir}")
    _log(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    start = time.time()
    exit_code = EXIT_CODES["unexpected_error"]

    try:
        dispatch = {
            "train":  run_train,
            "eval":   run_eval,
            "export": run_export,
            "deploy": run_deploy,
        }
        if mode not in dispatch:
            exit_code = _error(f"Unknown mode: {mode}", "invalid_args")
        else:
            exit_code = dispatch[mode](cfg, output_dir)
    except KeyboardInterrupt:
        _log("Interrupted by user.", "WARN")
        exit_code = EXIT_CODES["subprocess_failed"]
    except Exception as e:
        _log(f"Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        exit_code = EXIT_CODES["unexpected_error"]
    finally:
        elapsed = time.time() - start
        status  = "success" if exit_code == 0 else "failed"
        _write_manifest(
            output_dir, mode, cfg, status, exit_code, elapsed,
            extra={"repo_root": str(REPO_ROOT)})
        _log(f"Done. Status={status}, exit_code={exit_code}, "
             f"elapsed={elapsed:.1f}s")
        _log(f"Artifacts written to: {output_dir}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
