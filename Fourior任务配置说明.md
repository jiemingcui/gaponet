# Fourior 任务配置说明

## ✅ 已完成的工作

根据您的训练命令 `--task Isaac-Humanoid-Operator-Delta-Action-Fourior`，您**不需要创建新的 task 目录**，而是继续使用 `humanoid_operator` 任务。

我已经为您创建了以下 Fourior 相关的配置文件：

### 1. 环境配置文件
- **文件**: `source/sim2real/sim2real/tasks/humanoid_operator/humanoid_operator_env_cfg_fourior.py`
- **说明**: Fourior 机器人的环境配置，使用 Fourior 的机器人模型和关节名称

### 2. 环境实现文件
- **文件**: `source/sim2real/sim2real/tasks/humanoid_operator/humanoid_operator_env_fourior.py`
- **说明**: 基于 `humanoid_operator_env.py` 复制，使用 Fourior 的配置

### 3. 运行器配置
- **文件**: `source/sim2real/sim2real/tasks/humanoid_operator/agents/rsl_rl_operator_cfg.py`
- **添加**: `HumanoidOperatorFourierRunnerCfg` 类

### 4. 任务注册
- **文件**: `source/sim2real/sim2real/tasks/humanoid_operator/__init__.py`
- **状态**: ✅ 已注册 `Isaac-Humanoid-Operator-Delta-Action-Fourior` 任务

## 📋 配置详情

### 机器人配置
- **无负载版本**: `fourior_with_hand_fix` → 使用 `FOURIOR_CFG_WITH_HAND_FIX`
- **带负载版本**: `fourior_with_hand_fix_payload` → 使用 `FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD`（默认）

### 关节名称
- 使用 `ROBOT_JOINT_NAME_DICT_URDF_FOURIOR` 字典（已在 `joint_names.py` 中定义）
- 注意：Fourior 的关节命名与 H1 略有不同（如 `elbow_pitch_joint` vs `elbow_joint`）

## ⚠️ 需要注意的事项

### 1. USD 文件（必需）
根据 `fourior.py` 的配置，系统需要以下 USD 文件：
- `usds/fourior/gr3v2_2_2.usd` - 无负载版本
- `usds/fourior_payload/gr3v2_2_2_payload.usd` - 带负载版本（如果使用）

**如何创建**: 在 Isaac Sim 中导入 URDF 文件并保存为 USD 格式。

### 2. 传感器位置配置
在 `humanoid_operator_env_cfg_fourior.py` 中，我已经将部分关节名称从 `elbow_joint` 改为 `elbow_pitch_joint`（Fourior 的命名），但您可能需要根据实际需求进一步调整 `sensors_positions` 配置。

### 3. 参考身体链接
配置中使用的 `reference_body = "torso_link"`，如果 Fourior 机器人的躯干链接名称不同，需要修改。

### 4. 动作数据
确保动作数据文件格式正确，并放置在：
- `motions/motion_amass/edited_27dof/test.npz`

## 🚀 使用方式

您的训练命令已经可以正常使用：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Humanoid-Operator-Delta-Action-Fourior \
  --num_envs=4080 \
  --max_iterations 100000 \
  --experiment_name Sim2Real \
  --letter amass \
  --run_name delta_action_fourior_payload \
  --device cuda \
  env.mode=train \
  --headless
```

## 📁 文件结构

```
humanoid_operator/
├── __init__.py                          ✅ 已注册 Fourior 任务
├── humanoid_operator_env.py            (原始 H1 版本)
├── humanoid_operator_env_fourior.py    ✅ 新建（Fourior 版本）
├── humanoid_operator_env_cfg.py        (原始 H1 版本)
├── humanoid_operator_env_cfg_fourior.py ✅ 新建（Fourior 配置）
└── agents/
    └── rsl_rl_operator_cfg.py          ✅ 已添加 HumanoidOperatorFourierRunnerCfg
```

## 🔍 验证步骤

1. **检查文件是否存在**:
   ```bash
   ls -la gaponet/source/sim2real/sim2real/tasks/humanoid_operator/*fourior*
   ```

2. **检查 USD 文件**:
   ```bash
   ls -la gaponet/source/sim2real_assets/sim2real_assets/usds/fourior/
   ```

3. **运行训练命令**:
   如果 USD 文件已创建，可以直接运行训练命令。

## 🆘 如果遇到问题

1. **导入错误**: 检查 `sim2real_assets` 是否正确导出了 Fourior 配置
2. **关节名称错误**: 检查 `joint_names.py` 中的 Fourior 关节名称是否正确
3. **USD 文件缺失**: 这是最常见的问题，必须创建 USD 文件
4. **传感器配置**: 如果传感器相关错误，检查 `sensors_positions` 中的关节名称是否与 Fourior 匹配

