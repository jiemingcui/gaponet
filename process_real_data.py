"""
真实机器人数据处理脚本

本脚本用于处理从真实机器人（sage1 和 sage2）采集的数据，主要功能包括：
1. 从 CSV 文件读取机器人控制、状态和事件数据
2. 根据事件（MOTION_START、DISABLE）对齐时间戳
3. 将数据重采样到统一频率（50Hz 或 100Hz）
4. 提取关节位置、速度、力矩数据
5. 生成 npz 格式的训练数据文件
6. 生成可视化图表（可选）

数据目录结构：
- sage1/real/real_Xkg_Y/real/gr3v2_2_2/amass/<motion_name>/
  - control.csv: 控制命令数据
  - state_motor.csv: 电机状态数据
  - event.csv: 事件时间戳（MOTION_START、DISABLE 等）
"""

import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
from scipy import signal
from scipy.spatial import distance
import re
import pandas as pd
import shutil

# 需要过滤掉的手部关节名称（这些关节不在目标关节列表中）
HAND_JOINT_NAMES = ["left_hand_joint", "right_hand_joint"]


class RobotDataProcessor:
    """
    机器人数据处理器
    
    负责加载和处理单个动作的机器人数据，包括：
    - 从 CSV 文件读取控制命令和电机状态
    - 解析关节配置
    - 处理时间戳和单位转换
    - 提取关节位置、速度、力矩数据
    """

    def __init__(self, robot_name, motion_source, motion_name, is_simulation=True, file_root=None):
        """
        初始化数据处理器
        
        Args:
            robot_name: 机器人名称（如 'gr3v2_2_2'），用于查找配置文件
            motion_source: 动作数据源名称
            motion_name: 动作名称（相对路径）
            is_simulation: 是否为仿真数据（True=仿真，False=真实数据）
            file_root: 文件根目录（对于 sage1/sage2，这是 amass 目录）
        """
        self.robot_name = robot_name
        self.motion_source = motion_source
        self.motion_name = motion_name
        # 对于 sage1/sage2：file_root 已经是 amass 目录，motion_name 是相对于它的路径
        if file_root:
            self.file_path = os.path.join(file_root, motion_name)
        else:
            self.file_path = f"{motion_source}/{motion_name}"
        self.is_simulation = is_simulation

        # 设置数据格式
        if is_simulation:
            self.use_radians = True  # 仿真数据使用弧度
            self.use_seconds = True  # 仿真数据时间单位是秒
        else:
            self.use_radians = True  # 真实数据也使用弧度（需要从度转换）
            self.use_seconds = False  # 真实数据时间戳是微秒，需要转换

        # 从配置文件加载所有关节列表
        self.joint_config = self._load_joint_config()
        # 加载该动作感兴趣的关节列表（可能被 joint_list.txt 过滤）
        self.joint_list = self._load_joint_list()
        # 加载机器人数据（控制命令和电机状态）
        self.data = self._load_robot_data(["control", "state_motor"])

    def _load_joint_config(self):
        """
        从 YAML 配置文件加载关节列表
        
        Returns:
            tuple: 关节名称元组
        
        Raises:
            FileNotFoundError: 如果找不到配置文件
            ValueError: 如果配置文件格式无效
        """
        base_dir = Path(__file__).parent / "configs"
        # 优先使用机器人特定的配置文件
        primary = base_dir / f"{self.robot_name}_joints.yaml"
        # 回退到通用配置文件
        fallback = base_dir / "gr3v2_2_2_joints.yaml"

        config_path = primary if primary.is_file() else fallback
        if not config_path.is_file():
            raise FileNotFoundError(f"Joint config not found: {primary} nor fallback: {fallback}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "joints" not in config:
            raise ValueError(f"Invalid joints config: {config_path}")

        return tuple(config["joints"])

    def _load_joint_list(self):
        """
        加载关节列表
        
        优先从动作目录下的 joint_list.txt 文件读取，如果不存在则使用配置文件中的关节列表
        
        Returns:
            tuple: 关节名称元组（提取路径的最后一部分作为关节名）
        """
        joint_list_path = Path(f"{self.file_path}/joint_list.txt")

        if joint_list_path.is_file():
            # 如果存在 joint_list.txt，使用它（可能包含路径，只取最后一部分）
            with open(joint_list_path) as file:
                return tuple(line.strip().split("/")[-1] for line in file.readlines())
        else:
            # 否则使用配置文件中的关节列表
            return tuple(joint.strip().split("/")[-1] for joint in self.joint_config)

    def _load_robot_data(self, data_types):
        """
        加载和处理机器人数据文件
        
        Args:
            data_types: 要加载的数据类型列表（如 ["control", "state_motor"]）
        
        Returns:
            dict: 包含原始数据和处理后的 DOF 数据的字典
                - raw_data: 原始 CSV 数据
                - dof_command: 处理后的控制命令数据（关节位置）
                - dof_state: 处理后的电机状态数据（关节位置、速度、力矩）
        """
        robot_data = {}
        initial_time = float("inf")  # 用于找到所有数据文件中的最早时间戳

        # 读取每种类型的数据
        for data_type in data_types:
            data = pd.read_csv(f"{self.file_path}/{data_type}.csv")

            # 如果需要，转换时间单位（从微秒转换为秒）
            if not self.use_seconds:
                data["timestamp"] = data["timestamp"] / 1e6

            # 记录最早的时间戳（用于后续对齐）
            initial_time = min(initial_time, data["timestamp"][0])
            robot_data[data_type] = data

        # 计算相对时间戳
        for data in robot_data.values():
            # 相对于最早时间戳的时间（从 0 开始）
            data["time_since_zero"] = data["timestamp"] - initial_time
            # 与上一次采样的时间间隔
            data["time_since_last"] = data["timestamp"].diff()

        # 处理 DOF（自由度）数据
        # 控制命令：只包含位置信息
        dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"], self.use_radians)
        # 电机状态：包含位置、速度、力矩
        dof_state = self._process_dof_data(
            robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"], self.use_radians
        )

        return {"raw_data": robot_data, "dof_command": dof_command, "dof_state": dof_state}

    @property
    def real_event(self):
        """
        读取事件时间戳
        
        Returns:
            dict: 事件名称到时间戳（秒）的映射，例如：
                {'MOTION_START': 123.456, 'DISABLE': 456.789}
        """
        event = pd.read_csv(f"{self.file_path}/event.csv")
        # 将时间戳从微秒转换为秒，并转换为字典
        return dict(zip(event["event"], event["timestamp"] / 1e6))

    @property
    def raw_data(self):
        """返回原始数据（未处理的 CSV 数据）"""
        return self.data["raw_data"]

    @property
    def dof_command(self):
        """返回处理后的控制命令数据（关节位置命令）"""
        return self.data["dof_command"]

    @property
    def dof_state(self):
        """返回处理后的电机状态数据（关节位置、速度、力矩）"""
        return self.data["dof_state"]

    def _preprocess_string(self, value):
        """
        预处理字符串，为标识符添加引号以便安全解析
        
        Args:
            value: 包含列表字符串的值（如 "[1, 2, 3]"）
        
        Returns:
            str: 预处理后的字符串
        """
        pattern = r"\b([a-zA-Z_][a-zA-Z_0-9]*)\b"
        return re.sub(pattern, r"'\1'", value)

    def _safe_str_to_list(self, value):
        """
        安全地将字符串转换为列表
        
        Args:
            value: 字符串形式的列表（如 "[1, 2, 3]"）
        
        Returns:
            list: 解析后的列表，如果失败返回错误字符串
        """
        try:
            preprocessed_value = self._preprocess_string(value)
            return ast.literal_eval(preprocessed_value)
        except (ValueError, SyntaxError):
            return "Invalid list string"

    def _safe_deg2rad(self, x, i):
        """
        安全地将度数转换为弧度
        
        Args:
            x: 列表或数组
            i: 索引
        
        Returns:
            float: 转换后的弧度值，如果不是数字则返回原值
        """
        if pd.api.types.is_number(x[i]):
            return np.deg2rad(x[i])
        return x[i]

    def _process_dof_data(self, df, joint_list, keys=["positions"], is_rad=False):
        """
        处理 DOF（自由度）数据，将字符串列表转换为单独的列
        
        Args:
            df: 包含字符串列表列的 DataFrame（如 positions="[1, 2, 3]"）
            joint_list: 关节名称列表
            keys: 要处理的键列表（如 ["positions", "velocities", "torques"]）
            is_rad: 数据是否已经是弧度（True=弧度，False=需要从度转换）
        
        Returns:
            pd.DataFrame: 处理后的 DataFrame，每个关节的每个键都有单独的列
                例如：positions_left_hip_joint, velocities_left_hip_joint 等
        """
        df_copy = df.copy()

        for k in keys:
            # 将字符串列表转换为实际的列表
            df_copy[k] = df_copy[k].apply(self._safe_str_to_list)

            # 为每个关节创建单独的列
            key_df = pd.DataFrame(index=df.index)
            for i, j in enumerate(joint_list):
                # 如果是位置或速度且不是弧度，需要从度转换为弧度
                if not is_rad and k in ["positions", "velocities"]:
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: self._safe_deg2rad(x, i))
                else:
                    # 直接提取值（已经是弧度或力矩不需要转换）
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: x[i])

            # 将新列添加到 DataFrame，删除原来的字符串列
            df_copy = pd.concat([df_copy.drop(k, axis=1), key_df], axis=1)
        return df_copy

    @property
    def joints(self):
        return self.joint_list


class SkipMotionError(Exception):
    """
    跳过动作异常
    
    当动作数据缺少必要的事件（如 MOTION_START、DISABLE）时抛出此异常
    """
    pass


class RobotDataComparator:
    """
    机器人数据比较器和处理器
    
    用于处理多个动作的真实机器人数据，主要功能：
    1. 搜索并发现所有动作数据目录
    2. 对齐时间戳（基于 MOTION_START 和 DISABLE 事件）
    3. 重采样到统一频率（50Hz 或 100Hz）
    4. 提取并保存为 npz 格式
    5. 生成可视化图表
    """

    def __init__(self, robot_name, motion_source, motion_names, valid_joints_file, result_folder: str, sample_dt: int):
        """
        初始化数据比较器
        
        Args:
            robot_name: 机器人名称（如 'gr3v2_2_2'），用于查找配置文件和输出组织
            motion_source: 动作数据源名称（通常与 robot_name 相同）
            motion_names: 动作名称过滤（"*" 表示所有，或使用子串过滤）
            valid_joints_file: 可选的关节掩码文件路径（用于过滤关节）
            result_folder: 数据根目录（包含所有动作数据的目录）
            sample_dt: 采样时间间隔（本实现中按频率自动设定，此参数未使用）
        """

        self._robot_name = robot_name
        self._motion_source = motion_source
        self._result_folder = result_folder
        self._valid_joints_list = self._load_valid_joints(valid_joints_file, self._robot_name)
        self._valid_joints_list = [path.split("/")[-1] for path in self._valid_joints_list]

        self._use_all_joints = (len(self._valid_joints_list) == 0)

        # Search robot directory, return motion relative paths and frequencies
        self._motion_names, self._motion_freqs = self._process_motion_names(
            motion_names, self._result_folder, self._robot_name, self._motion_source
        )
        if len(self._motion_names) == 0:
            raise ValueError("Can't get any motion_name")

        self._sample_dt = sample_dt

        # Lazy loading cache: load data for each motion on demand
        self._motion_cache: dict[str, RobotDataProcessor] = {}

    def _get_real_data(self, motion_name: str) -> RobotDataProcessor:
        """Load and cache data processor for a motion on demand"""
        if motion_name not in self._motion_cache:
            # For sage1/sage2 structure: result_folder is already the amass directory
            # motion_name is the relative path from result_folder, so file_path should be result_folder/motion_name
            self._motion_cache[motion_name] = RobotDataProcessor(
                self._robot_name,
                self._robot_name,
                motion_name,
                is_simulation=False,
                file_root=self._result_folder,  # result_folder is already the base directory
            )
        return self._motion_cache[motion_name]

    def _get_joints_for_motion(self, motion_name):
        sim_data = self._get_real_data(motion_name)

        if self._use_all_joints:
            selected_joints = sim_data.joints
            print(f"Using all {len(selected_joints)} joints for motion {motion_name}")
        else:
            sim_joints = sim_data.joints
            selected_joints = list(set(self._valid_joints_list).intersection(set(sim_joints)))
            if not selected_joints:
                selected_joints = sim_joints
                print(f"Using all {len(selected_joints)} joints for motion {motion_name} (mask had no common joints)")
            else:
                print(f"Using {len(selected_joints)} joints from mask intersection for motion {motion_name}")

        return [j for j in selected_joints if j not in HAND_JOINT_NAMES]

    def _load_valid_joints(self, valid_joints_file=None, robot_name=None):
        """
        Load valid joints file
        """
        if valid_joints_file and os.path.exists(valid_joints_file):
            with open(valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        default_valid_joints_file = f"configs/{robot_name}_valid_joints.txt"
        if os.path.exists(default_valid_joints_file):
            with open(default_valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        print(f"Warning: No joints mask file found for '{robot_name}'. Using all joints.")
        return []

    def _process_motion_names(self, motion_names_arg, result_folder, robot_name, motion_source):
        """
        递归搜索所有动作文件夹
        
        查找包含必要 CSV 文件（control.csv, event.csv, state_motor.csv）的目录，
        并从路径中检测采样频率（50Hz 或 100Hz）。
        
        Args:
            motion_names_arg: 动作名称过滤参数（"*" 或子串）
            result_folder: 数据根目录
            robot_name: 机器人名称（未使用）
            motion_source: 动作数据源（未使用）
        
        Returns:
            tuple: (motion_names, motion_freqs)
                - motion_names: 动作相对路径列表（如 "motion1", "motion2"）
                - motion_freqs: 字典，键为动作名称，值为频率（50 或 100）
        """
        base_dir = result_folder
        if not os.path.exists(base_dir):
            raise ValueError(f"Real results directory does not exist: {base_dir}")

        need_files = {"control.csv", "event.csv", "state_motor.csv"}
        motion_names = []
        motion_freqs = {}

        # Simple substring filter (can be "*" or specific substring)
        filter_token = None if (motion_names_arg is None or motion_names_arg == "*") else motion_names_arg.strip()

        hz_pattern = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)

        for dirpath, dirnames, filenames in os.walk(base_dir):
            files = set(filenames)
            if need_files.issubset(files):
                rel = os.path.relpath(dirpath, base_dir).replace("\\", "/")
                if filter_token and filter_token not in rel:
                    continue

                # Look for 50Hz/100Hz in path segments, default to 50Hz if not found
                freq = 50  # Default frequency for real data
                for part in rel.split("/"):
                    m = hz_pattern.search(part)
                    if m:
                        cand = int(m.group(1))
                        if cand in (50, 100):
                            freq = cand
                            break

                motion_names.append(rel)
                motion_freqs[rel] = freq

        print(f"Found {len(motion_names)} motions. All with frequency (default 50Hz if not specified).")
        return motion_names, motion_freqs

    def adjust_real_data_timing(self, dataframe, start_delay, end_time):
        """
        调整真实机器人数据的时间戳
        
        根据 MOTION_START 事件对齐时间戳，并过滤到 MOTION_START 到 DISABLE 之间的数据。
        
        Args:
            dataframe: 包含时间戳的 DataFrame
            start_delay: MOTION_START 事件的时间戳（秒）
            end_time: DISABLE 事件的时间戳（秒）
        
        Returns:
            pd.DataFrame: 调整后的 DataFrame，时间从 0 开始，到 (end_time - start_delay) 结束
        """
        df_adjusted = dataframe.copy()

        # 调整时间戳（减去 MOTION_START 时间，使动作开始时间为 0）
        for time_col in ["timestamp", "time_since_zero"]:
            if time_col in df_adjusted.columns:
                df_adjusted[time_col] -= start_delay

        # 移除时间戳为负的数据（动作开始之前的数据）
        if "time_since_zero" in df_adjusted.columns:
            mask = (df_adjusted["time_since_zero"] <= 0)
            if mask.any():
                first_non_positive_index = mask.idxmax()
                df_adjusted = df_adjusted.iloc[first_non_positive_index:].reset_index(drop=True)

        # 过滤时间范围：只保留 MOTION_START 到 DISABLE 之间的数据
        df_filtered = df_adjusted[df_adjusted["time_since_zero"] >= 0]
        max_time = end_time - start_delay

        return df_filtered[df_filtered["time_since_zero"] <= max_time]

    def _resample_waveform(self, df: pd.DataFrame, key: str, timestamp: np.ndarray) -> pd.DataFrame:
        """
        使用线性插值将波形重采样到新的时间戳数组
        
        Args:
            df: 包含原始数据的 DataFrame
            key: 要重采样的列名（如 "positions_left_hip_joint"）
            timestamp: 新的时间戳数组（均匀间隔）
        
        Returns:
            pd.DataFrame: 重采样后的数据，包含 timestamp 和 value 列
        """
        return pd.DataFrame({"timestamp": timestamp, "value": np.interp(timestamp, df["time_since_zero"], df[key])})

    def align_data(self, motion_name, joint_name, data_type="positions"):
        """
        对齐真实数据用于比较和保存
        
        根据 MOTION_START 和 DISABLE 事件对齐时间戳，并重采样到统一频率。
        
        Args:
            motion_name: 动作名称（相对路径）
            joint_name: 关节名称
            data_type: 数据类型（"positions", "velocities", "torques"）
        
        Returns:
            tuple: 
                - 如果 data_type == "positions": (aligned_real, aligned_real_cmd)
                - 否则: (aligned_real,)
                每个 DataFrame 包含对齐后的时间戳和数值
        """
        real_data = self._get_real_data(motion_name)
        ev = real_data.real_event  # Event name dictionary (e.g., contains 'MOTION_START', 'DISABLE')
        # Strictly require key events to exist; otherwise throw skip exception
        if "MOTION_START" not in ev or "DISABLE" not in ev:
            missing = []
            if "MOTION_START" not in ev:
                missing.append("MOTION_START")
            if "DISABLE" not in ev:
                missing.append("DISABLE")
            raise SkipMotionError(f"Missing events: {', '.join(missing)}")
        real_delay = ev["MOTION_START"]
        real_end = ev["DISABLE"]

        real_df = self.adjust_real_data_timing(real_data.dof_state, real_delay, real_end)
        real_cmd = self.adjust_real_data_timing(real_data.dof_command, real_delay, real_end)

        key = f"{data_type}_{joint_name}"

        # Use sampling interval selected by frequency parsed from this motion
        dt = 1.0 / float(self._motion_freqs[motion_name])
        max_time = real_df["time_since_zero"].max()
        timestamp = np.arange(dt, max_time, dt)

        aligned_real = self._resample_waveform(real_df, key, timestamp)
        if data_type == "positions":
            aligned_real_cmd = self._resample_waveform(real_cmd, key, timestamp)
            return aligned_real, aligned_real_cmd
        else:
            return aligned_real

    def analyze_all_data(self, output_dir, if_plot=True):
        """
        分析所有动作数据并生成输出文件
        
        对每个动作：
        1. 对齐时间戳（基于 MOTION_START 和 DISABLE 事件）
        2. 重采样到统一频率（50Hz 或 100Hz）
        3. 提取所有目标关节的位置、速度、力矩数据
        4. 保存为 npz 文件（用于训练）
        5. 生成可视化图表（可选）
        
        Args:
            output_dir: 输出目录（npz 文件保存位置）
            if_plot: 是否生成可视化图表（True=生成，保存到 plots/ 目录）
        """
        # Create output root directory
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"The provided output_dir '{output_dir}' exists but is not a directory.")
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Progress bar helper
        total_motions = len(self._motion_names)
        print(f"Detected {total_motions} motions in total.")

        def _print_progress(i: int, n: int, width: int = 30):
            i = max(0, min(i, n))
            filled = int(width * (i / n)) if n else width
            bar = "█" * filled + "-" * (width - filled)
            print(f"\rProcessing progress [{bar}] {i}/{n}", end="", flush=True)

        _print_progress(0, total_motions)

        # 目标关节列表（Fourior 机器人的所有关节）
        # 这些关节的数据将被提取并保存到 npz 文件中
        target_joints = [
            # "left_shoulder_pitch_joint",
            # "left_shoulder_roll_joint",
            # "left_shoulder_yaw_joint",
            # "left_elbow_pitch_joint",
            # "left_wrist_pitch_joint",
            # "right_shoulder_pitch_joint",
            # "right_shoulder_roll_joint",
            # "right_shoulder_yaw_joint",
            # "right_elbow_pitch_joint",
            # "right_wrist_pitch_joint"
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

        # Process each motion separately
        for idx, motion_name in enumerate(self._motion_names, start=1):
            print(f"\n[{idx}/{total_motions}] Processing motion: {motion_name}")
            freq = self._motion_freqs[motion_name]
            safe_motion = motion_name.replace("/", "_")

            # npz output directory: output_files/<robot>/<motion_frequency>/
            motion_out_dir = os.path.join(output_dir, f"{safe_motion}_{freq}Hz")
            os.makedirs(motion_out_dir, exist_ok=True)

            # Image output directory: plots/<robot>/<motion_frequency>/ (parallel to output_files)
            plot_out_dir = os.path.join("plots", self._robot_name, f"{safe_motion}_{freq}Hz")
            os.makedirs(plot_out_dir, exist_ok=True)

            try:
                # Filter: only keep target joints that actually exist in the data (load on demand)
                joints_available = self._get_real_data(motion_name).joints
                joints = [j for j in target_joints if j in joints_available]
                if not joints:
                    print(f"  Warning: no target joints found in motion {motion_name}, skip.")
                    raise SkipMotionError("no target joints available")

                # Collect all three types of data for all joints
                pos_list, pos_cmd_list, vel_list, tau_list = [], [], [], []
                for joint_name in joints:
                    # positions + command
                    aligned_real_pos, aligned_real_pos_cmd = self.align_data(motion_name, joint_name, "positions")
                    pos_vals = aligned_real_pos.iloc[:, 1].to_numpy()
                    pos_cmd_vals = aligned_real_pos_cmd.iloc[:, 1].to_numpy()

                    # velocities
                    aligned_real_vel = self.align_data(motion_name, joint_name, "velocities")
                    vel_vals = aligned_real_vel.iloc[:, 1].to_numpy()

                    # torques
                    aligned_real_tau = self.align_data(motion_name, joint_name, "torques")
                    tau_vals = aligned_real_tau.iloc[:, 1].to_numpy()

                    pos_list.append(pos_vals)
                    pos_cmd_list.append(pos_cmd_vals)
                    vel_list.append(vel_vals)
                    tau_list.append(tau_vals)

                # After aligning time axis, can directly stack
                real_dof_positions = np.stack(pos_list, axis=0)
                real_dof_positions_cmd = np.stack(pos_cmd_list, axis=0)
                real_dof_velocities = np.stack(vel_list, axis=0)
                real_dof_torques = np.stack(tau_list, axis=0)
                joint_sequence = np.array(joints, dtype=object)

                # Generate separate images for each joint (save to plots/<robot>/<motion_frequency>/)
                if if_plot:
                    x = np.arange(real_dof_positions.shape[1])
                    for i, jn in enumerate(joints):
                        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

                        axs[0].plot(x, real_dof_positions[i], label='position')
                        axs[0].plot(x, real_dof_positions_cmd[i], label='position command', linestyle='--', alpha=0.7)
                        axs[0].set_title(f'{jn} - position (solid) & position command (dashed)')
                        axs[0].legend()
                        axs[0].set_xlabel('Index'); axs[0].set_ylabel('Value')

                        axs[1].plot(x, real_dof_velocities[i], color='tab:blue', label='velocity')
                        axs[1].set_title(f'{jn} - velocity'); axs[1].set_xlabel('Index'); axs[1].set_ylabel('Value')

                        axs[2].plot(x, real_dof_torques[i], color='tab:blue', label='torque')
                        axs[2].set_title(f'{jn} - torque'); axs[2].set_xlabel('Index'); axs[2].set_ylabel('Value')

                        plt.tight_layout()
                        plt.savefig(os.path.join(plot_out_dir, f"{jn}.png"))
                        plt.close()

                # Save npz for this motion (only under output_files)
                npz_file_path = os.path.join(motion_out_dir, "motor_all_joints.npz")
                np.savez_compressed(
                    npz_file_path,
                    real_dof_positions=real_dof_positions,
                    real_dof_positions_cmd=real_dof_positions_cmd,
                    real_dof_velocities=real_dof_velocities,
                    real_dof_torques=real_dof_torques,
                    joint_sequence=joint_sequence,
                    motion_name=motion_name,
                    frequency=freq,
                )
                print(f"Saved npz to {npz_file_path}")
            except SkipMotionError as e:
                print(f"Skip motion due to missing events: {e}")
                # Delete created directories for this motion (npz/plots)
                if os.path.isdir(motion_out_dir):
                    shutil.rmtree(motion_out_dir, ignore_errors=True)
                if os.path.isdir(plot_out_dir):
                    shutil.rmtree(plot_out_dir, ignore_errors=True)
                _print_progress(idx, total_motions)
                continue

            # Update progress bar
            _print_progress(idx, total_motions)

        print()  # New line, end progress bar line


if __name__ == "__main__":
    """
    主函数：处理 sage1 和 sage2 真实机器人数据
    
    数据目录结构：
    - sage1/real/real_Xkg_Y/real/gr3v2_2_2/amass/<motion_name>/
      - control.csv: 控制命令数据
      - state_motor.csv: 电机状态数据
      - event.csv: 事件时间戳
    
    - sage2/real/real_Xkg_Y/real/gr3v2_2_2/amass/<motion_name>/
      - 同上
    
    输出结构：
    - output_files/sage1/real_Xkg_Y/<motion_name>_50Hz/motor_all_joints.npz
    - output_files/sage2/real_Xkg_Y/<motion_name>_50Hz/motor_all_joints.npz
    - plots/gr3v2_2_2/<motion_name>_50Hz/<joint_name>.png
    """
    
    parser = argparse.ArgumentParser(description="处理 sage1 和 sage2 真实机器人数据")
    parser.add_argument("--sage1", type=str, default=None, help="sage1 真实数据根目录")
    parser.add_argument("--sage2", type=str, default=None, help="sage2 真实数据根目录")
    parser.add_argument("--output", type=str, default="output_files", help="npz 文件输出目录")
    parser.add_argument("--robot-name", type=str, default="gr3v2_2_2", help="机器人名称（用于查找配置文件和输出组织）")
    args = parser.parse_args()

    # 处理 sage1 数据
    if os.path.exists(args.sage1):
        print("== 处理 sage1 真实数据 ==")
        # 查找所有 real_Xkg_Y 目录（如 real_0kg_0, real_1kg_0 等）
        sage1_dirs = [d for d in os.listdir(args.sage1) 
                     if os.path.isdir(os.path.join(args.sage1, d)) and d.startswith("real_")]
        
        for real_dir in sage1_dirs:
            # 构建数据根目录路径：sage1/real/real_Xkg_Y/real/gr3v2_2_2/amass
            real_data_root = os.path.join(args.sage1, real_dir, "real", args.robot_name, "amass")
            if not os.path.exists(real_data_root):
                print(f"  跳过 {real_dir}: 数据路径不存在")
                continue
            
            print(f"  处理 {real_dir}...")
            data_comparator = RobotDataComparator(
                robot_name=args.robot_name,
                motion_source=args.robot_name,
                motion_names="*",  # 处理所有动作
                valid_joints_file=None,  # 不使用关节掩码
                result_folder=real_data_root,
                sample_dt=None,  # 根据检测到的频率自动重采样（默认 50Hz）
            )
            # 输出目录：output_files/sage1/real_Xkg_Y/
            output_root = Path(args.output) / "sage1" / real_dir
            data_comparator.analyze_all_data(str(output_root), if_plot=True)
    else:
        print(f"Sage1 目录不存在: {args.sage1}")

    # 处理 sage2 数据（与 sage1 相同的流程）
    if os.path.exists(args.sage2):
        print("\n== 处理 sage2 真实数据 ==")
        # 查找所有 real_Xkg_Y 目录
        sage2_dirs = [d for d in os.listdir(args.sage2) 
                     if os.path.isdir(os.path.join(args.sage2, d)) and d.startswith("real_")]
        
        for real_dir in sage2_dirs:
            # 构建数据根目录路径：sage2/real/real_Xkg_Y/real/gr3v2_2_2/amass
            real_data_root = os.path.join(args.sage2, real_dir, "real", args.robot_name, "amass")
            if not os.path.exists(real_data_root):
                print(f"  跳过 {real_dir}: 数据路径不存在")
                continue
            
            print(f"  处理 {real_dir}...")
            data_comparator = RobotDataComparator(
                robot_name=args.robot_name,
                motion_source=args.robot_name,
                motion_names="*",  # 处理所有动作
                valid_joints_file=None,  # 不使用关节掩码
                result_folder=real_data_root,
                sample_dt=None,  # 根据检测到的频率自动重采样（默认 50Hz）
            )
            # 输出目录：output_files/sage2/real_Xkg_Y/
            output_root = Path(args.output) / "sage2" / real_dir
            data_comparator.analyze_all_data(str(output_root), if_plot=True)
    else:
        print(f"Sage2 目录不存在: {args.sage2}")