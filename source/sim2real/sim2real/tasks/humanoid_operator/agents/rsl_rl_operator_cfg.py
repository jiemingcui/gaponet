# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from typing import Literal, List
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class DeepONetActorCriticCfg:
    class_name: str = "DeepONetActorCritic"

    """Configuration for DeepONet Actor-Critic network"""
    # Branch network configuration
    branch_input_dims: List[int] = [400]  # Input dimensions for different resolutions
    branch_hidden_dim: int = 256  # Hidden dimension for branch networks
    
    # Trunk network configuration
    trunk_input_dim: int = 10#+10  # Input dimension for trunk network
    trunk_hidden_dims: List[int] = [128, 128, 128]  # Hidden dimensions for trunk network
    
    # Output configuration
    activation: str = "elu"  # Activation function
    
    # Critic network configuration
    critic_input_dim: int = 400+10+30+20+1+2+32  # Input dimension for critic network
    critic_hidden_dims: List[int] = [256, 128, 128]  # Hidden dimensions for critic network

    # Model network configuration
    model_input_dim: int = 10+30*4  # Input dimension for model network
    model_output_dim: int = 400  # Output dimension for model network
    model_hidden_dims: List[int] = [128, 128]  # Hidden dimensions for model network

    # Model history configuration
    model_history_length: int = 4  # Number of history steps to keep
    model_history_dim: int = 30
    model_pretrained_path: str = ""

@configclass
class HumanoidOperatorRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OperatorRunner"

    """Configuration for DeepONet PPO runner"""
    num_steps_per_env = 32
    num_steps_function = 1

    max_iterations = 120
    save_interval = 50
    replay_buffer_size = 40
    experiment_name = "humanoid_operator"
    empirical_normalization = True
    
    direct_sample_envs = True
    full_trajectory_sampling = True
    
    # DeepONet policy configuration
    policy = DeepONetActorCriticCfg() # type: ignore
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
    
    # Logger configuration
    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    wandb_project: str = "humanoid-deeponet-direct"

    # model learning configuration
    model_based_sensor = True
    model_replay_buffer_size = 50
    model_learning_epochs = 300
    model_learning_steps = 1
    model_learning_interval = 1000
    model_sample_iterations = 10

    # zero-shot transformation configuration
    retrain_sensor_only = False

    # augmentation configuration
    randomize_dynamics = True

    # evaluation configuration
    eval_after_training = False

@configclass
class HumanoidOperatorVanillaRunnerCfg(HumanoidOperatorRunnerCfg):
    class_name = "OperatorVanillaRunner"