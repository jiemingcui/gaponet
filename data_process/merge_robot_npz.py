"""
Robot Motion Data Merger

This script merges scattered robot motion data files (.npz format) into standardized datasets 
for machine learning and analysis. Key features include:

1. Data Collection & Classification:
   - Recursively searches for .npz files in specified directories
   - Automatically detects sampling frequencies (50Hz/100Hz) from file paths or content
   - Classifies and collects data by frequency

2. Data Standardization:
   - Aligns time axis to (T, D) format (time first, features second)
   - Normalizes joint sequence formats
   - Extracts payload information from file paths (e.g., "real_3kg_1" -> 3) or motion names

3. Joint Extraction:
   - Extracts only 10 target joints from source data:
     left/right_shoulder_pitch/roll/yaw_joint, left/right_elbow_pitch_joint,
     left/right_wrist_pitch_joint
   - Output data dimension is 10 (one per target joint)

4. Output Generation:
   - Produces standardized merged files: merged_50Hz.npz and merged_100Hz.npz
   - Output format aligns with test.npz structure for downstream processing

Usage: python merge_robot_npz.py -r output_files -o merged_npz
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

def detect_freq_from_path(path: Path) -> Optional[int]:
    """Parse 50Hz/100Hz from various directory levels in the path"""
    hz_pat = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)
    for part in path.parts:
        m = hz_pat.search(part)
        if m:
            try:
                val = int(m.group(1))
                if val in (50, 100):
                    return val
            except ValueError:
                pass
    return None

def find_npz_files(root: Path) -> List[Path]:
    """Recursively search for all .npz files under root directory"""
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".npz"):
                files.append(Path(dirpath) / f)
    return files

# New: Extract payload (kg) as integer (rounded) from string or path; return None if failed
# Pattern 1: real_Xkg_Y format (e.g., "real_3kg_1" -> 3)
_REAL_KG_PAT = re.compile(r'real[_\s-]*(\d+(?:\.\d+)?)\s*kg', re.IGNORECASE)
# Pattern 2: General Xkg format (e.g., "5kg", "1.5kg")
_KG_PAT = re.compile(r'(\d+(?:\.\d+)?)\s*kg(?=[/_\-\s]|$)', re.IGNORECASE)

def extract_payload_int_kg(name: Optional[str], file_path: Optional[Path] = None) -> Optional[int]:
    """
    Extract payload (kg) as integer from file path or name string.
    Priority: 1) file path (real_Xkg_Y format), 2) name string (Xkg format)
    
    Args:
        name: String to search (e.g., motion_name)
        file_path: File path to search (e.g., Path object)
    
    Returns:
        Integer payload value, or None if not found
    """
    # First try: extract from file path (real_Xkg_Y format)
    if file_path is not None:
        path_str = str(file_path)
        m = _REAL_KG_PAT.search(path_str)
        if m:
            try:
                val = float(m.group(1))
                return int(round(val))
            except Exception:
                pass
    
    # Second try: extract from name string (general Xkg format)
    if name:
        s = str(name)
        m = _KG_PAT.search(s)
        if m:
            try:
                val = float(m.group(1))
                return int(round(val))
            except Exception:
                pass
    
    return None

# New: Normalize joint_sequence to string list
def normalize_joint_sequence(js_any: Any) -> Optional[List[str]]:
    if js_any is None:
        return None
    try:
        # Common forms: np.ndarray(dtype=object or <U..), list, tuple
        if isinstance(js_any, np.ndarray):
            if js_any.ndim == 0:
                v = js_any.item()
                if isinstance(v, (list, tuple, np.ndarray)):
                    return [str(x) for x in list(v)]
                elif isinstance(v, (str, np.str_)):
                    return [str(v)]
                else:
                    return None
            # 1D string array
            if js_any.ndim == 1 and (js_any.dtype == object or np.issubdtype(js_any.dtype, np.str_)):
                out = [str(x) for x in js_any.tolist()]
                return out if all(isinstance(x, str) for x in out) else None
            # Other forms try tolist
            out = list(js_any.tolist())
            return [str(x) for x in out] if all(isinstance(x, (str, np.str_)) for x in out) else None
        if isinstance(js_any, (list, tuple)):
            out = [str(x) for x in js_any]
            return out if all(isinstance(x, str) for x in out) else None
        if isinstance(js_any, (str, np.str_)):
            return [str(js_any)]
    except Exception:
        return None
    return None

# New: Convert any 2D array to (T, D); prioritize judgment based on D_guess
def to_time_first(a: np.ndarray, D_guess: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    if D_guess is not None:
        if a.shape[1] == D_guess:
            return a
        if a.shape[0] == D_guess:
            return a.T
    # Heuristic: if first dimension is smaller (channel count usually < time length), consider it (D, T), need transpose
    return a.T if a.shape[0] <= a.shape[1] else a

# Target joints: only 10 joints as specified
# These 10 joints will be extracted and used as joint_sequence
TARGET_JOINTS = [
    # "left_shoulder_pitch_joint",
    # "left_shoulder_roll_joint",
    # "left_shoulder_yaw_joint",
    # "left_elbow_pitch_joint",
    # "left_wrist_pitch_joint",
    # "right_shoulder_pitch_joint",
    # "right_shoulder_roll_joint",
    # "right_shoulder_yaw_joint",
    # "right_elbow_pitch_joint",
    # "right_wrist_pitch_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint" 
]
TARGET_JOINTS_10 = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch_joint",
]

def extract_10_joints(series_list, src_names):
    """
    Extract only the 10 target joints from (T, D_src) sequences to (T, 10).
    Only columns matching TARGET_JOINTS are kept, in the order specified.
    """
    name_to_src = {name: i for i, name in enumerate(src_names)}
    out = []
    missing = []
    for a in series_list:
        if not isinstance(a, np.ndarray) or a.ndim != 2:
            raise ValueError(f"Expected 2D array, got: {type(a)}, shape={getattr(a, 'shape', None)}")
        T, _ = a.shape
        b = np.zeros((T, len(TARGET_JOINTS)), dtype=a.dtype)
        for dst_idx, name in enumerate(TARGET_JOINTS):
            if name in name_to_src:
                src_idx = name_to_src[name]
                b[:, dst_idx] = a[:, src_idx]
            else:
                # Record missing names, only count first time
                missing.append(name)
        out.append(b)
    if missing:
        # Only prompt once (deduplicated)
        miss_set = sorted(set(missing))
        print(f"[Warning] Following joints missing in source joint_sequence, will remain 0: {miss_set}")
    return out

def main():
    parser = argparse.ArgumentParser(description="Merge npz files with same frequency under output_files to merged_npz (aligned with test.npz format).")
    parser.add_argument("-r", "--root", type=str, default="output_files", help="npz root directory (default: output_files)")
    parser.add_argument("-o", "--out", type=str, default="merged_npz", help="output directory (default: merged_npz)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Root directory does not exist: {root}")
        return

    files = find_npz_files(root)
    print(f"Found npz files: {len(files)}")

    # Changed to only collect keys needed for test.npz alignment
    buckets: Dict[int, Dict[str, List[Any]]] = {
        50: {"positions": [], "positions_cmd": [], "velocities": [], "torques": [],
             "payloads": [], "joint_candidates": []},
        100: {"positions": [], "positions_cmd": [], "velocities": [], "torques": [],
              "payloads": [], "joint_candidates": []},
    }

    used = 0
    skipped = 0
    total = len(files)
    for i, f in enumerate(files, start=1):
        freq = detect_freq_from_path(f)
        if freq is None:
            # Try to read frequency field from file
            try:
                with np.load(f, allow_pickle=True) as npzf:
                    freq_inside = int(npzf["frequency"]) if "frequency" in npzf.files else None
                    if freq_inside in (50, 100):
                        freq = freq_inside
            except Exception:
                pass

        if freq not in (50, 100):
            skipped += 1
            print(f"[{i}/{total}] Skip (unable to identify frequency): {f}")
            continue

        try:
            with np.load(f, allow_pickle=True) as npzf:
                # Required keys
                req = ["real_dof_positions", "real_dof_positions_cmd", "real_dof_velocities", "real_dof_torques"]
                if not all(k in npzf.files for k in req):
                    skipped += 1
                    print(f"[{i}/{total}] Skip (missing key fields): {f}")
                    continue

                # joint_sequence candidates
                jlist = normalize_joint_sequence(npzf["joint_sequence"]) if "joint_sequence" in npzf.files else None
                if jlist:
                    buckets[freq]["joint_candidates"].append(jlist)

                D_guess = len(jlist) if jlist else None

                # Convert to (T, D)
                pos = to_time_first(npzf["real_dof_positions"], D_guess=D_guess)
                pos_cmd = to_time_first(npzf["real_dof_positions_cmd"], D_guess=D_guess)
                vel = to_time_first(npzf["real_dof_velocities"], D_guess=D_guess)
                tor = to_time_first(npzf["real_dof_torques"], D_guess=D_guess)

                # Parse payloads (from file path's real_Xkg_Y or motion_name's Xkg)
                motion_name = str(npzf["motion_name"]) if "motion_name" in npzf.files else None
                payload = extract_payload_int_kg(motion_name, file_path=f)
                if payload is None:
                    # If unable to parse, set to -1 and prompt
                    payload = -1
                    print(f"[{i}/{total}] Warning: Unable to extract payload from path or motion_name, set to -1: {f}")

                # Append
                buckets[freq]["positions"].append(pos)
                buckets[freq]["positions_cmd"].append(pos_cmd)
                buckets[freq]["velocities"].append(vel)
                buckets[freq]["torques"].append(tor)
                buckets[freq]["payloads"].append(payload)
                used += 1
                print(f"[{i}/{total}] Collected: {f} -> {freq}Hz, shapes pos={pos.shape}, vel={vel.shape}, tor={tor.shape}, payload={payload}")
        except Exception as e:
            skipped += 1
            print(f"[{i}/{total}] Skip (read failed): {f} ({e})")

    # Save merged results for both frequencies (aligned with test.npz)
    for freq in (50, 100):
        data = buckets[freq]
        N = len(data["positions"])
        if N == 0:
            print(f"{freq}Hz has no available samples, skip saving.")
            continue

        def to_object_array(seq: List[Any]) -> np.ndarray:
            arr = np.empty(len(seq), dtype=object)
            for i, v in enumerate(seq):
                arr[i] = v
            return arr

        # Get source joint names from the first candidate (for extraction)
        src_js: List[str] = None
        candidates: List[Tuple[str, ...]] = [tuple(js) for js in data["joint_candidates"] if js]
        if candidates:
            # Use the most frequent candidate as source joint names
            cnt = {}
            for c in candidates:
                cnt[c] = cnt.get(c, 0) + 1
            src_js = list(max(cnt.items(), key=lambda x: x[1])[0])
        else:
            # If no candidates, try to infer from first sample's dimension
            D = data["positions"][0].shape[1]
            src_js = [f"joint_{i}" for i in range(D)]
            print(f"{freq}Hz no joint_sequence found, using placeholder names: D={D}")

        # Extract only the 10 target joints from the data
        print(f"{freq}Hz extracting {len(TARGET_JOINTS)} target joints from source data...")
        data["positions"]      = extract_10_joints(data["positions"],      src_js)
        data["positions_cmd"]  = extract_10_joints(data["positions_cmd"],  src_js)
        data["velocities"]     = extract_10_joints(data["velocities"],     src_js)
        data["torques"]        = extract_10_joints(data["torques"],        src_js)
        final_D = len(TARGET_JOINTS)

        save_path = out_dir / f"merged_{freq}Hz.npz"
        np.savez_compressed(
            save_path,
            real_dof_positions=to_object_array(data["positions"]),
            real_dof_positions_cmd=to_object_array(data["positions_cmd"]),
            real_dof_velocities=to_object_array(data["velocities"]),
            real_dof_torques=to_object_array(data["torques"]),
            joint_sequence=np.asarray(TARGET_JOINTS_10, dtype=object),  # Use the 10 target joints
            payloads=np.asarray(data["payloads"], dtype=np.int64),
        )
        print(f"Saved merged file: {save_path} (samples: {N}, joint names: {len(TARGET_JOINTS)}, data columns: {final_D})")

    print(f"Complete. Used: {used}, Skipped: {skipped}, Total: {len(files)}")

if __name__ == "__main__":
    main()