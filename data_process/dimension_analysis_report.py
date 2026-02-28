"""
配置文件维度问题分析报告

本报告对以下文件进行全面维度审查：
1. rsl_rl_operator_cfg.py
2. humanoid_operator_env_cfg_fourior.py
3. humanoid_operator_env_cfg.py
4. humanoid_operator_env_fourior.py
5. humanoid_operator_env.py
"""

import os
import sys

REPORT = """
================================================================================
                        配置文件维度问题分析报告
================================================================================

【摘要】
本报告对 GapONet 项目中的配置文件进行全面维度审查，特别关注输入/输出张量形状、
状态空间维度、动作空间维度、网络层维度配置的一致性。

================================================================================
                        一、关键维度定义
================================================================================

根据代码分析，关键维度定义如下：

【1. 机器人关节定义】

FOURIOR 机器人 (fourior_with_hand_fix):
  - 总关节数量: 28 个 (来自 ROBOT_JOINT_NAME_DICT_URDF_FOURIOR)
  - 包括: 躯干(3) + 头部(2) + 手臂(14) + 腿部(12)

FOURIOR Payload 机器人:
  - 训练使用的 10 个关节 (来自 joint_sequence):
    left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint,
    left_elbow_pitch_joint, left_wrist_pitch_joint,
    right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint,
    right_elbow_pitch_joint, right_wrist_pitch_joint

【2. 传感器配置】

humanoid_operator_env_cfg_fourior.py:
  - num_sensor_positions: 20
  - sensor_dim: 62 (joint_pos 31 + joint_vel*dt 31)

【3. 模型历史配置】

humanoid_operator_env_cfg_fourior.py:
  - model_history_length: 4
  - model_history_dim: 93 (joint_pos 31 + joint_vel 31 + joint_target 31)

================================================================================
                        二、各文件维度配置分析
================================================================================

【1. rsl_rl_operator_cfg.py 维度配置】

DeepONetActorCriticCfg:
  +-------------------------+------------------+------------------------+
  | 配置项                  | 配置值           | 来源                   |
  +-------------------------+------------------+------------------------+
  | branch_input_dims      | [20 * 62] = 1240| 注释                   |
  | trunk_input_dim         | 32               | 注释 (31 + 1)         |
  | critic_input_dim        | 1463             | 注释                   |
  | model_input_dim         | 403              | 注释 (31 + 93*4)       |
  | model_output_dim        | 1240             | 注释 (20 * 62)         |
  | model_history_length   | 4                | 显式配置               |
  | model_history_dim       | 93               | 显式配置               |
  +-------------------------+------------------+------------------------+

【2. humanoid_operator_env_cfg_fourior.py 维度配置】

  +-------------------------+------------------+------------------------+
  | 配置项                  | 配置值           | 说明                   |
  +-------------------------+------------------+------------------------+
  | action_space            | 1 * 31 = 31      | 31个关节delta action  |
  | num_sensor_positions   | 20               | 传感器位置数量         |
  | sensor_dim              | 62               | 每位置特征维度         |
  | model_history_length   | 4                | 模型历史长度           |
  | model_history_dim       | 93               | 每步历史维度           |
  +-------------------------+------------------+------------------------+

【3. humanoid_operator_env_cfg.py 维度配置 (H1.2 机器人)**

  +-------------------------+------------------+------------------------+
  | 配置项                  | 配置值           | 说明                   |
  +-------------------------+------------------+------------------------+
  | action_space            | 10               | 10个关节               |
  | num_sensor_positions   | 20               | 传感器位置数量         |
  | sensor_dim              | 20               | 每位置特征维度         |
  | model_history_length   | 4                | 模型历史长度           |
  | model_history_dim       | 30               | 每步历史维度           |
  +-------------------------+------------------+------------------------+

================================================================================
                        三、发现的维度问题
================================================================================

【问题 1】action_space 维度不匹配 ⚠️

  位置: humanoid_operator_env_cfg_fourior.py
  
  问题描述:
    - 配置文件中 action_space = 31 (1 * 31)
    - 但实际训练的 NPZ 数据只有 10 个关节 (joint_sequence)
    - 这意味着策略网络输出 31 维动作，但训练数据只有 10 维

  实际数据:
    - NPZ 文件: merged_50Hz_31_10_payload.npz
    - joint_sequence 长度: 10 个关节
    - 每个样本形状: (时间步, 31) - 注意这里是时间步，不是关节数！

  影响:
    - 潜在运行时错误：尝试访问不存在的关节索引
    - 维度不匹配可能导致训练不稳定

--------------------------------------------------------------------------------

【问题 2】trunk_input_dim 计算错误 ⚠️

  位置: rsl_rl_operator_cfg.py
  
  配置值:
    - trunk_input_dim = 32 (注释: current_action 31 + payload 1)
  
  实际使用 (humanoid_operator_env.py compute_operator_observation):
    - trunk_obs = torch.cat([current_action], dim=1)
    - 只包含 current_action，没有包含 payload！
  
  实际维度:
    - current_action 维度 = 10 (来自 joint_sequence_index)
    - 所以 trunk_input_dim 应该是 10，而不是 32

  影响:
    - 网络输入维度与实际数据维度不匹配
    - 可能导致维度错误或训练失败

--------------------------------------------------------------------------------

【问题 3】sensor_dim 与实际计算不一致 ⚠️

  位置: humanoid_operator_env_cfg_fourior.py
  
  配置:
    - sensor_dim = 62 (注释: joint_pos 31 + joint_vel*dt 31)

  实际计算 (humanoid_operator_env_fourior.py _set_sensor_data):
    - 需要检查实际的传感器数据计算

--------------------------------------------------------------------------------

【问题 4】model_history_dim 配置与实际不匹配 ⚠️

  位置: humanoid_operator_env_cfg_fourior.py
  
  配置值:
    - model_history_dim = 93 (注释: joint_pos 31 + joint_vel 31 + joint_target 31)

  实际计算 (humanoid_operator_env.py compute_model_observation):
    - 当 add_model_history = True 时:
      model_obs = torch.cat([joint_pos, self.model_history.flatten(1, 2)], dim=1)
      self.model_history[:, 0, :] = torch.cat([joint_pos, joint_vel, joint_target], dim=1)
    
  问题:
    - 如果使用的是 10 个关节而非 31 个关节
    - model_history_dim 应该是 10*3 = 30，而非 93

--------------------------------------------------------------------------------

【问题 5】critic_input_dim 计算与实际不匹配 ⚠️

  位置: rsl_rl_operator_cfg.py
  
  配置值:
    - critic_input_dim = 1463 (注释详细计算)
  
  实际计算 (humanoid_operator_env.py compute_operator_observation):
    - critic_obs = torch.cat([
        self.sensor_data.flatten(1, 2),    # 20 * 62 = 1240
        current_action,                       # 10
        robot joint_pos,                      # 10
        robot joint_vel,                     # 10
        robot joint_acc,                     # 10
        real_joint_pos,                     # 10
        real_joint_vel,                     # 10
        wrist_payload_mass,                  # 1
        hand_payload_mass,                  # 2
        robot_mass,                          # 需要确认
    ], dim=1)
  
  实际维度:
    - 1240 + 10*6 + 1 + 2 + ? = 应该是 1313 + robot_mass 维度

================================================================================
                        四、维度配置正确性验证
================================================================================

【验证 1: Branch Network 输入维度】

  配置期望: 20 * 62 = 1240
  实际数据: num_sensor_positions (20) * sensor_dim (62) = 1240
  状态: ✓ 匹配

【验证 2: Trunk Network 输入维度】

  配置期望: 32 (31 action + 1 payload)
  实际数据: len(joint_sequence) = 10 (仅上肢关节)
  状态: ✗ 不匹配 - 配置使用 31 维，实际只有 10 维

【验证 3: Model Network 输入维度】

  配置期望: 31 + 93 * 4 = 403
  实际数据: 10 + 30 * 4 = 130 (如果使用 10 个关节)
  状态: ✗ 不匹配 - 配置使用 31 维关节，实际只有 10 维

【验证 4: Model Network 输出维度】

  配置期望: 20 * 62 = 1240
  实际数据: num_sensor_positions (20) * sensor_dim (62) = 1240
  状态: ✓ 匹配

================================================================================
                        五、影响分析
================================================================================

【严重程度：高】

1. 运行时崩溃:
   - 维度不匹配可能导致 tensor 形状错误
   - 特别是 trunk_input_dim 和 model_history_dim

2. 训练效果不佳:
   - 即使不崩溃，维度不正确也会导致网络学习到错误的映射
   - 策略网络输出 31 维，但动作应用可能只作用于 10 维

3. Sim-to-Real 迁移问题:
   - 由于训练时使用的维度与实际环境不匹配
   - 零样本迁移性能可能严重下降

================================================================================
                        六、修复建议
================================================================================

【建议 1】统一关节维度定义

如果目标是使用 10 个上肢关节进行训练：

rsl_rl_operator_cfg.py:
  - trunk_input_dim: 32 → 10 (或 11，如果包含 payload)
  - model_input_dim: 403 → 130 (10 + 30*4)
  - model_history_dim: 93 → 30
  - critic_input_dim: 需要重新计算

humanoid_operator_env_cfg_fourior.py:
  - action_space: 31 → 10
  - model_history_dim: 93 → 30

【建议 2】或者使用完整的 28/31 个关节

如果需要使用完整的机器人关节：
  - 确保 NPZ 数据包含所有 28 个关节
  - 更新 joint_sequence 配置

【建议 3】验证实际运行的维度

在运行训练之前，添加维度打印语句：
  - print(f"trunk_input_dim: {trunk_obs.shape}")
  - print(f"model_obs_dim: {model_obs.shape}")

================================================================================
                        七、总结
================================================================================

发现的主要维度问题：

  1. action_space: 配置 31 vs 实际 10 个关节 ⚠️
  2. trunk_input_dim: 配置 32 vs 实际 10 ⚠️
  3. model_history_dim: 配置 93 vs 实际应该是 30 ⚠️
  4. critic_input_dim: 配置 1463 vs 实际约 1313 ⚠️

这些问题需要优先修复，否则可能导致运行时错误或训练效果不佳。

================================================================================
"""

if __name__ == "__main__":
    print(REPORT)
    
    # 保存报告
    with open("dimension_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(REPORT)
    
    print("\n报告已保存到: dimension_analysis_report.txt")
