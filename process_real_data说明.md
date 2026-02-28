# process_real_data.py 命令说明

## 命令概述

```bash
python process_real_data.py \
    --sage1 sage1/real \
    --sage2 sage2/real \
    --output output_files \
    --robot-name gr3v2_2_2
```

## 命令作用

这个命令用于**处理从真实机器人采集的数据**，将原始的 CSV 文件转换为训练所需的 npz 格式文件。

## 详细功能

### 1. 数据输入

脚本会从以下目录结构读取数据：

```
sage1/real/
└── real_Xkg_Y/              # 不同负载配置（如 real_0kg_0, real_1kg_0）
    └── real/
        └── gr3v2_2_2/
            └── amass/
                └── <motion_name>/    # 每个动作一个目录
                    ├── control.csv      # 控制命令数据
                    ├── state_motor.csv  # 电机状态数据
                    └── event.csv        # 事件时间戳（MOTION_START, DISABLE）

sage2/real/
└── （同上结构）
```

### 2. 数据处理流程

对每个动作目录，脚本会执行以下步骤：

#### 步骤 1: 读取 CSV 文件
- 读取 `control.csv`（控制命令）
- 读取 `state_motor.csv`（电机状态：位置、速度、力矩）
- 读取 `event.csv`（事件时间戳）

#### 步骤 2: 时间戳对齐
- 从 `event.csv` 中提取 `MOTION_START` 和 `DISABLE` 事件
- 将时间戳对齐到 `MOTION_START`（动作开始时间设为 0）
- 只保留 `MOTION_START` 到 `DISABLE` 之间的数据

#### 步骤 3: 数据转换
- 将时间戳从微秒转换为秒
- 将角度从度转换为弧度（如果需要）
- 将字符串列表（如 `"[1, 2, 3]"`）解析为实际的数组

#### 步骤 4: 重采样
- 检测采样频率（从路径中查找 "50Hz" 或 "100Hz"，默认 50Hz）
- 使用线性插值将数据重采样到统一频率
- 确保所有关节数据的时间轴对齐

#### 步骤 5: 提取目标关节数据
提取以下 31 个关节的数据：
- **腿部**（12 个）：左右髋、膝、踝关节
- **躯干**（3 个）：腰部偏航、横滚、俯仰
- **头部**（2 个）：头部偏航、俯仰
- **手臂**（14 个）：左右肩、肘、腕关节

#### 步骤 6: 生成输出文件

**npz 文件**（用于训练）：
```
output_files/
├── sage1/
│   └── real_Xkg_Y/
│       └── <motion_name>_50Hz/
│           └── motor_all_joints.npz
└── sage2/
    └── real_Xkg_Y/
        └── <motion_name>_50Hz/
            └── motor_all_joints.npz
```

npz 文件包含：
- `real_dof_positions`: 关节位置（形状：[关节数, 时间步数]）
- `real_dof_positions_cmd`: 关节位置命令
- `real_dof_velocities`: 关节速度
- `real_dof_torques`: 关节力矩
- `joint_sequence`: 关节名称序列
- `motion_name`: 动作名称
- `frequency`: 采样频率

**可视化图表**（可选）：
```
plots/
└── gr3v2_2_2/
    └── <motion_name>_50Hz/
        ├── left_hip_pitch_joint.png
        ├── right_hip_pitch_joint.png
        └── ...（每个关节一个图表）
```

每个图表包含 3 个子图：
- 位置和位置命令（实线和虚线）
- 速度
- 力矩

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--sage1` | sage1 真实数据根目录 | `sage1/real` |
| `--sage2` | sage2 真实数据根目录 | `sage2/real` |
| `--output` | npz 文件输出目录 | `output_files` |
| `--robot-name` | 机器人名称（用于查找配置文件） | `gr3v2_2_2` |

## 使用场景

1. **数据预处理**：将原始采集的 CSV 数据转换为训练格式
2. **数据验证**：通过可视化图表检查数据质量
3. **数据对齐**：确保所有动作数据的时间轴对齐
4. **格式统一**：将不同频率的数据统一到 50Hz 或 100Hz

## 注意事项

1. **必需文件**：每个动作目录必须包含：
   - `control.csv`
   - `state_motor.csv`
   - `event.csv`（必须包含 `MOTION_START` 和 `DISABLE` 事件）

2. **缺少事件**：如果某个动作缺少必要事件，该动作会被跳过

3. **关节过滤**：脚本会自动过滤掉手部关节（`left_hand_joint`, `right_hand_joint`）

4. **配置文件**：需要 `configs/gr3v2_2_2_joints.yaml` 文件来定义关节列表

## 输出示例

运行命令后，您会看到类似输出：

```
== 处理 sage1 真实数据 ==
找到 5 个动作。所有动作频率（未指定时默认 50Hz）。
  处理 real_0kg_0...
[1/5] 处理动作: motion1
使用所有 31 个关节处理动作 motion1
保存 npz 到 output_files/sage1/real_0kg_0/motion1_50Hz/motor_all_joints.npz
[2/5] 处理动作: motion2
...
处理进度 [████████████████████████████] 5/5

== 处理 sage2 真实数据 ==
...
```

## 后续使用

生成的 npz 文件可以用于：
1. 训练强化学习模型
2. 数据分析和可视化
3. 仿真到真实的迁移学习

