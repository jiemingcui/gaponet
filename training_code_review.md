# 训练代码检查报告

## 训练命令
```bash
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action-Fourior \
  --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real \
  --letter amass --run_name delta_action_fourior_payload --device cuda \
  env.mode=train --headless
```

## 一、Reward 设计机制

### 1.1 Reward 计算流程

#### 主要文件
- `source/sim2real/sim2real/tasks/humanoid_operator/humanoid_operator_env_fourior.py`

#### Reward 函数：`_get_rewards()` (Line 465-480)

```python
def _get_rewards(self) -> torch.Tensor:
    # Get per-joint tracking errors (shape: num_envs, 31)
    per_joint_tracking_error = self._reward_tracking()
    
    # Calculate total reward (mean across joints) - this is used for training
    total_tracking_error = torch.mean(per_joint_tracking_error, dim=1)  # shape: (num_envs,)
    total_reward = - ((36 / (2 * torch.pi)) ** 2) * total_tracking_error
    
    # Convert per-joint tracking error to per-joint reward (same scaling as total reward)
    per_joint_reward = - ((36 / (2 * torch.pi)) ** 2) * per_joint_tracking_error  # shape: (num_envs, 31)
    
    # Accumulate per-joint rewards for logging (will be logged at episode end)
    self.per_joint_reward_buffer += per_joint_reward
    self.per_joint_reward_count += 1
    
    return total_reward
```

#### 核心 Tracking Reward：`_reward_tracking()` (Line 1165-1188)

```python
def _reward_tracking(self):
    """
    Calculate tracking error for each joint.
    
    Returns:
        torch.Tensor: Per-joint tracking error of shape (num_envs, num_joints)
                     where num_joints is the length of joint_sequence_index (31)
    """
    # robot current state
    robot_dof_positions = self.robot.data.joint_pos     # shape: (num_envs, num_dofs)
    robot_dof_velocities = self.robot.data.joint_vel    # shape: (num_envs, num_dofs)

    # sampled state from MotionLoader
    real_dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]   # shape: (num_envs, num_dofs)
    real_dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices] # shape: (num_envs, num_dofs)
    
    joint_index = self._motion_loader.joint_sequence_index   # shape: (31,)

    # calculate per-joint errors (don't take mean across joints)
    position_diff = (robot_dof_positions[:, joint_index] - real_dof_positions[:, joint_index]) ** 2     # shape: (num_envs, 31)
    velocity_diff = (robot_dof_velocities[:, joint_index] - real_dof_velocities[:, joint_index]) ** 2   # shape: (num_envs, 31)

    # Return per-joint tracking error (shape: num_envs, 31)
    return position_diff + velocity_diff * 1e-2
```

### 1.2 Reward 设计要点

1. **Tracking Error 计算**
   - 位置误差：`(robot_pos - real_pos)²`
   - 速度误差：`(robot_vel - real_vel)² * 1e-2` (速度误差权重为0.01)
   - 每个关节独立计算误差，形状为 `(num_envs, 31)`

2. **Reward 缩放**
   - 缩放因子：`-((36 / (2 * π))²) ≈ -32.8`
   - 将 tracking error 转换为负 reward
   - 设计意图：将角度误差（rad）转换为度数误差（deg）的平方，然后取负

3. **总 Reward 计算**
   - 对31个关节的 tracking error 取平均：`torch.mean(per_joint_tracking_error, dim=1)`
   - 然后应用缩放因子得到总 reward
   - **用于训练的总 reward 是所有关节的平均 tracking error**

4. **Per-Joint Reward 记录**
   - 同时计算每个关节的 reward（用于日志记录）
   - 累积在 `per_joint_reward_buffer` 中
   - 在 episode 结束时记录到 CSV 文件

### 1.3 Reward 在 Step 中的使用 (Line 988-990)

```python
rewards = self._get_rewards()
rewards = rewards * unfinished_motion  # 只对未完成的motion给予reward
dones = ~unfinished_motion
```

- Reward 只在 motion 未完成时给予（`unfinished_motion` 为 True）
- Motion 完成时 reward 为 0

### 1.4 潜在问题检查

✅ **正常点：**
- Reward 设计合理：基于 tracking error，鼓励机器人跟踪参考轨迹
- 速度误差权重较小（1e-2），主要关注位置跟踪
- 使用负 reward，符合最小化误差的目标

⚠️ **需要注意：**
- 缩放因子 `(36 / (2π))²` 的设计意图需要确认（可能是将 rad 转换为 deg 的平方）
- 速度误差权重 `1e-2` 可能需要调优
- 没有其他 reward 项（如平滑度、能量消耗等），完全依赖 tracking error

## 二、参数更新机制

### 2.1 训练流程

#### 主要文件
- `source/sim2real/sim2real/rsl_rl/runners/operator_runner.py`
- `source/sim2real/sim2real/tasks/humanoid_operator/agents/rsl_rl_operator_cfg.py`

### 2.2 PPO 算法配置

#### 配置文件：`rsl_rl_operator_cfg.py` (Line 79-93)

```python
algorithm = RslRlPpoAlgorithmCfg(
    class_name="PPO",
    value_loss_coef=1.0,              # Value loss 系数
    use_clipped_value_loss=True,      # 使用 clipped value loss
    clip_param=0.2,                   # PPO clip 参数
    entropy_coef=0.0,                 # 熵系数（无探索奖励）
    num_learning_epochs=5,            # 每次更新进行5轮学习
    num_mini_batches=4,               # 每个 epoch 使用4个 mini-batch
    learning_rate=1.0e-4,             # 学习率：0.0001
    schedule="adaptive",              # 自适应学习率调度
    gamma=0.99,                       # 折扣因子
    lam=0.95,                         # GAE lambda 参数
    desired_kl=0.008,                 # 期望 KL 散度
    max_grad_norm=1.0,                # 梯度裁剪阈值
)
```

### 2.3 训练循环：`learn()` 函数

#### 数据收集阶段 (Line 221-383)

```python
def learn(self, num_learning_iterations: int, **kwargs):
    for it in range(start_iter, num_learning_iterations):
        # 1. 数据收集
        for step in range(self.num_steps_per_env):  # 32 steps
            # 获取动作
            actions = self.alg.act(...)
            # 环境步进
            obs, rewards, dones, infos = self.env.step(actions)
            # 处理环境步进
            self.alg.process_env_step(rewards, dones, infos)
        
        # 2. 计算回报
        if self.training_type == "rl":
            self.alg.compute_returns(privileged_obs)  # GAE 计算
        
        # 3. 更新策略
        loss_dict = self.alg.update()  # PPO 更新
```

#### 关键参数
- `num_steps_per_env = 32`：每个环境收集32步数据
- `num_steps_function = 1`：每个函数采样1步（用于 sensor model）

### 2.4 参数更新机制

#### PPO Update 流程（在 `self.alg.update()` 中）

1. **数据准备**
   - 使用收集的 `num_steps_per_env * num_envs` 步数据
   - 数据包括：observations, actions, rewards, values, log_probs

2. **多轮学习** (`num_learning_epochs=5`)
   - 每个 epoch 将数据分成 `num_mini_batches=4` 个 mini-batch
   - 对每个 mini-batch 进行梯度更新

3. **Loss 计算**
   - **Policy Loss (Clipped)**: 
     ```python
     ratio = new_log_prob / old_log_prob
     clipped_ratio = clip(ratio, 1-clip_param, 1+clip_param)
     policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
     ```
   - **Value Loss (Clipped)**:
     ```python
     value_loss = (value - returns)²
     if use_clipped_value_loss:
         value_loss = clip(value_loss, ...)
     ```
   - **Total Loss**:
     ```python
     total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
     ```

4. **梯度更新**
   - 计算梯度：`loss.backward()`
   - 梯度裁剪：`torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)`
   - 参数更新：`optimizer.step()`

5. **自适应学习率**
   - `schedule="adaptive"`：根据 KL 散度调整学习率
   - 如果 KL 散度 > `desired_kl`，降低学习率
   - 如果 KL 散度 < `desired_kl`，提高学习率

### 2.5 网络架构

#### DeepONet Actor-Critic

**配置文件**：`rsl_rl_operator_cfg.py` (Line 11-55)

- **Branch Network**:
  - 输入：`sensor_data.flatten(1, 2)` → 1240 维 (20 sensor positions × 62 dims)
  - 隐藏层：256 维
  - 输出：`action_dim * 16` 维

- **Trunk Network**:
  - 输入：`current_action(31) + payload(1)` → 32 维
  - 隐藏层：`[128, 128, 128]`
  - 输出：`action_dim * 16` 维

- **Critic Network**:
  - 输入：1463 维（sensor_data + robot_state + real_state + payload + mass）
  - 隐藏层：`[256, 128, 128]`
  - 输出：1 维（value）

- **Model Network** (Sensor Model):
  - 输入：`model_obs_dim`（由环境决定）
  - 输出：1240 维（sensor data）

### 2.6 优化器配置

- **Policy Optimizer**: Adam, `lr=1.0e-4`
- **Sensor Model Optimizer**: Adam, `lr=1.0e-4` (Line 57 in operator_runner.py)

### 2.7 潜在问题检查

✅ **正常点：**
- PPO 配置合理：clip_param=0.2, 5个学习轮次，4个mini-batch
- 使用梯度裁剪防止梯度爆炸
- 自适应学习率调度有助于稳定训练
- 无熵奖励（entropy_coef=0.0）适合确定性任务

⚠️ **需要注意：**
- **学习率可能偏小**：`1.0e-4` 对于100k迭代可能需要更长时间收敛
- **KL 散度目标**：`desired_kl=0.008` 需要监控实际 KL 散度
- **Value Loss 系数**：`value_loss_coef=1.0` 可能需要调优
- **无探索机制**：`entropy_coef=0.0` 可能限制策略探索

## 三、训练数据流

### 3.1 数据收集流程

1. **环境初始化** (4080 个并行环境)
2. **每个迭代收集 32 步数据**
3. **数据包括**：
   - Observations (branch, trunk, critic)
   - Actions (31 维 delta action)
   - Rewards (tracking error based)
   - Values (critic 输出)
   - Log probabilities (action 的 log prob)

### 3.2 Reward 处理

- Reward 在 `step_operator()` 中计算
- 只对未完成的 motion 给予 reward
- Reward 累积在 replay buffer 中
- 用于计算 GAE (Generalized Advantage Estimation)

### 3.3 参数更新频率

- **每次迭代**：收集 32 步数据 → 计算回报 → 更新策略（5个epoch，每个4个mini-batch）
- **总更新次数**：100,000 次迭代
- **总步数**：100,000 × 32 × 4080 = 13,056,000,000 步

## 四、关键检查点

### 4.1 Reward 机制检查

- [x] Reward 计算正确：基于 tracking error
- [x] Reward 缩放合理：负 reward，鼓励最小化误差
- [x] Per-joint reward 记录：用于分析和调试
- [ ] 是否需要其他 reward 项（平滑度、能量等）？

### 4.2 参数更新检查

- [x] PPO 配置合理
- [x] 学习率设置存在
- [x] 梯度裁剪启用
- [x] 自适应学习率调度
- [ ] 学习率可能需要调优（当前 1e-4）
- [ ] KL 散度监控需要确认

### 4.3 网络架构检查

- [x] DeepONet 架构正确
- [x] 输入输出维度匹配
- [x] Sensor model 独立训练
- [ ] 网络容量是否足够？

## 五、建议

### 5.1 Reward 设计

1. **考虑添加平滑度奖励**（当前 `_reward_delta_smoothness` 被禁用）
2. **速度误差权重**：当前 `1e-2`，可能需要调整
3. **Reward 缩放**：确认 `(36/(2π))²` 的设计意图

### 5.2 参数更新

1. **学习率**：考虑使用学习率调度（warmup + decay）
2. **监控 KL 散度**：确保策略更新不会太大
3. **Value Loss 系数**：可能需要根据训练情况调整

### 5.3 训练监控

1. **Per-joint reward**：已实现，用于分析各关节表现
2. **Tracking error**：监控位置和速度误差
3. **Policy update**：监控 KL 散度、学习率变化

## 六、代码位置总结

### Reward 相关
- `humanoid_operator_env_fourior.py:465` - `_get_rewards()`
- `humanoid_operator_env_fourior.py:1165` - `_reward_tracking()`
- `humanoid_operator_env_fourior.py:1190` - `_reward_delta_smoothness()` (未使用)

### 参数更新相关
- `operator_runner.py:221` - `learn()` 训练循环
- `operator_runner.py:386` - `self.alg.update()` PPO 更新
- `rsl_rl_operator_cfg.py:79` - PPO 算法配置

### 网络架构
- `rsl_rl_operator_cfg.py:11` - DeepONet 配置
- `deeponet_actor_critic.py` - 网络实现





