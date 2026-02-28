# GapONet 代码结构文档

## 项目概述

GapONet 是一个基于 Isaac Lab 的人形机器人强化学习框架，支持 DeepONet、Transformer 和 MLP 架构的 sim-to-real 迁移。

## 目录结构

```
gaponet/
├── apps/                          # Isaac Sim 应用文件
│   ├── isaaclab.python.headless.kit
│   ├── isaaclab.python.headless.rendering.kit
│   ├── isaaclab.python.kit
│   ├── isaaclab.python.rendering.kit
│   └── isaaclab.python.xr.openxr.kit
│
├── outputs/                       # 训练输出日志（已忽略）
│   └── 2026-01-28/...
│
├── scripts/                      # 训练和演示脚本
│   ├── benchmarks/               # 性能基准测试
│   │   ├── benchmark_cameras.py
│   │   ├── benchmark_load_robot.py
│   │   ├── benchmark_non_rl.py
│   │   ├── benchmark_rlgames.py
│   │   ├── benchmark_rsl_rl.py
│   │   └── utils.py
│   │
│   ├── demos/                    # 示例演示
│   │   ├── sensors/              # 传感器演示
│   │   ├── arms.py               # 机械臂
│   │   ├── bipeds.py             # 双足机器人
│   │   ├── deformables.py        # 软体对象
│   │   ├── hands.py              # 机械手
│   │   ├── markers.py            # 标记
│   │   ├── multi_asset.py        # 多资产
│   │   ├── procedural_terrain.py # 程序化地形
│   │   ├── quadcopter.py         # 四旋翼
│   │   └── quadrupeds.py         # 四足机器人
│   │
│   ├── environments/             # 环境相关
│   │   ├── state_machine/        # 状态机环境
│   │   ├── teleoperation/        # 遥操作
│   │   ├── list_envs.py
│   │   ├── random_agent.py
│   │   └── zero_agent.py
│   │
│   ├── imitation_learning/        # 模仿学习
│   │   ├── isaaclab_mimic/
│   │   └── robomimic/
│   │
│   ├── real_deploy/              # 实际部署
│   │   └── goal_reaching_server.py
│   │
│   ├── reinforcement_learning/    # 强化学习（已忽略）
│   │   ├── ray/                  # Ray 分布式训练
│   │   ├── rl_games/
│   │   ├── rsl_rl/
│   │   ├── sb3/
│   │   └── skrl/
│   │
│   ├── tools/                    # 工具脚本
│   │   ├── blender_obj.py
│   │   ├── convert_mesh.py
│   │   ├── convert_mjcf.py
│   │   ├── convert_urdf.py
│   │   ├── pretrained_checkpoint.py
│   │   ├── record_demos.py
│   │   └── replay_demos.py
│   │
│   └── tutorials/                # 教程
│       ├── 00_sim/
│       ├── 01_assets/
│       ├── 02_scene/
│       ├── 03_envs/
│       ├── 04_sensors/
│       └── 05_controllers/
│
├── source/                       # 核心源代码
│   ├── isaaclab/                 # Isaac Lab 框架
│   │   ├── config/
│   │   │   └── extension.toml
│   │   └── isaaclab/
│   │       ├── actuators/        # 执行器
│   │       ├── app/              # 应用启动器
│   │       ├── assets/           # 资产（机器人/对象）
│   │       │   ├── articulation/  # 关节式资产（机器人）
│   │       │   ├── deformable_object/
│   │       │   ├── rigid_object/ # 刚体对象
│   │       │   └── rigid_object_collection/
│   │       ├── controllers/      # 控制器
│   │       │   ├── differential_ik.py    # 差分逆运动学
│   │       │   ├── joint_impedance.py    # 关节阻抗
│   │       │   ├── operational_space.py # 操作空间控制
│   │       │   └── rmp_flow.py            # RMP 流程
│   │       ├── devices/         # 输入设备
│   │       │   ├── gamepad/      # 手柄
│   │       │   ├── keyboard/     # 键盘
│   │       │   ├── openxr/       # XR 设备
│   │       │   └── spacemouse/   # 空间鼠标
│   │       ├── envs/             # 环境
│   │       │   ├── mdp/          # 马尔可夫决策过程
│   │       │   │   ├── actions/  # 动作
│   │       │   │   ├── commands/ # 命令
│   │       │   │   ├── recorders/# 记录器
│   │       │   │   ├── curriculums.py
│   │       │   │   ├── events.py
│   │       │   │   ├── observations.py
│   │       │   │   ├── rewards.py
│   │       │   │   └── terminations.py
│   │       │   ├── ui/           # UI
│   │       │   └── *env*.py      # 环境基类
│   │       ├── managers/         # 管理器
│   │       ├── markers/          # 可视化标记
│   │       ├── scene/            # 场景
│   │       ├── sensors/          # 传感器
│   │       │   ├── camera/       # 相机
│   │       │   ├── contact_sensor/
│   │       │   ├── frame_transformer/
│   │       │   ├── imu/
│   │       │   └── ray_caster/
│   │       ├── sim/              # 仿真
│   │       │   ├── converters/   # 格式转换
│   │       │   ├── schemas/
│   │       │   ├── spawners/     # 生成器
│   │       │   └── utils.py
│   │       ├── terrains/         # 地形
│   │       ├── ui/               # UI 组件
│   │       └── utils/            # 工具函数
│   │
│   ├── isaaclab_assets/          # Isaac Lab 资产
│   │
│   ├── isaaclab_mimic/           # 模仿学习扩展
│   │
│   ├── isaaclab_rl/              # 强化学习扩展
│   │
│   ├── isaaclab_tasks/          # 任务环境
│   │
│   ├── sim2real/                 # 核心项目代码
│   │   ├── sim2real/
│   │   │   ├── rsl_rl/           # RSL-RL 强化学习
│   │   │   │   ├── algorithms/   # 算法
│   │   │   │   ├── modules/      # 网络模块
│   │   │   │   │   ├── ActorCriticTransformer.py  # Transformer Actor-Critic
│   │   │   │   │   ├── deeponet_actor_critic.py   # DeepONet Actor-Critic
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── networks/     # 网络架构
│   │   │   │   │   ├── lnn_sensor_model.py
│   │   │   │   │   └── multi_res_branch_net.py
│   │   │   │   └── runners/      # 训练运行器
│   │   │   │       ├── base_runner.py
│   │   │   │       ├── operator_runner.py
│   │   │   │       └── operator_vanilla_runner.py
│   │   │   │
│   │   │   └── tasks/            # 任务定义
│   │   │       ├── humanoid_operator/  # 操作员任务
│   │   │       │   ├── agents/          # Agent 配置
│   │   │       │   │   ├── rsl_rl_operator_cfg.py
│   │   │       │   │   └── rsl_rl_ppo_cfg.py
│   │   │       │   ├── humanoid_operator_env.py
│   │   │       │   ├── humanoid_operator_env_cfg.py
│   │   │       │   ├── humanoid_operator_env_fourior.py
│   │   │       │   ├── humanoid_operator_env_cfg_fourior.py
│   │   │       │   ├── operator_helper.py
│   │   │       │   ├── motions/         # 运动数据
│   │   │       │   │   ├── joint_names.py
│   │   │       │   │   └── motion_motor_loader.py
│   │   │       │   └── utils/
│   │   │       │
│   │   │       └── humanoid_amass/ # AMASS 任务
│   │   │           ├── agents/
│   │   │           ├── amass_delta_action_env.py
│   │   │           ├── amass_delta_action_env_cfg.py
│   │   │           └── motions/
│   │   │
│   │   └── build/                # 构建输出（已忽略）
│   │
│   └── sim2real_assets/          # 机器人资产
│
├── .gitignore                    # Git 忽略规则
├── README.md                     # 项目说明
├── environment.yml               # Conda 环境配置
├── pyproject.toml               # Python 项目配置
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装脚本
├── setup.sh                     # 安装脚本（Linux/Mac）
├── isaaclab.sh                  # Isaac Lab 启动脚本
├── isaaclab.bat                 # Isaac Lab 启动脚本（Windows）
└── LICENCE                      # 许可证
```

## 核心模块说明

### 1. 强化学习模块 (`source/sim2real/sim2real/rsl_rl/`)

#### 网络架构 (modules/)
- **ActorCriticTransformer.py**: Transformer 架构的 Actor-Critic
- **deeponet_actor_critic.py**: DeepONet 架构的 Actor-Critic
- **lnn_sensor_model.py**: 传感器模型
- **multi_res_branch_net.py**: 多分辨率分支网络

#### 运行器 (runners/)
- **base_runner.py**: 基础运行器
- **operator_runner.py**: 操作员任务运行器
- **operator_vanilla_runner.py**: 标准操作员运行器

### 2. 任务环境 (`source/sim2real/sim2real/tasks/`)

#### humanoid_operator
人形机器人操作员任务，包含：
- 环境配置 (`humanoid_operator_env_cfg.py`)
- 环境实现 (`humanoid_operator_env.py`)
- Agent 配置 (`agents/rsl_rl_operator_cfg.py`)
- 运动数据加载 (`motions/motion_motor_loader.py`)

#### humanoid_amass
基于 AMASS 数据集的人形机器人任务

### 3. Isaac Lab 框架 (`source/isaaclab/isaaclab/`)

核心仿真框架，包含：
- **assets/**: 机器人/对象资产
- **controllers/**: 控制器
- **envs/**: 强化学习环境
- **sensors/**: 传感器
- **sim/**: 仿真底层

## 训练脚本

主要训练入口位于 `scripts/rsl_rl/`:
- **train.py**: 训练脚本
- **play.py**: 模型评估/播放
- **cli_args.py**: 命令行参数

## 数据文件

需要从外部下载：
1. **sim2real_assets**: 机器人资产
2. **test_data**: 测试数据 (NPZ 格式)
3. **checkpoint**: 训练好的模型权重

## 依赖

- Isaac Sim 4.5.0+
- Python 3.10+
- CUDA
- Isaac Lab
- PyTorch
- RSL-RL
