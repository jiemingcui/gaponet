# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic import ActorCritic


class ActorCriticTransformer(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        
        actor_hidden_dims=[128, 512],          # d_model, dim_feedforward
        critic_hidden_dims=[128, 512],         # d_model, dim_feedforward
        nhead=4,                               # Number of attention heads
        num_layers=2,                          # Number of transformer layers
        activation="gelu",                     # Recommended activation function
        
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # Initialize nn.Module directly instead of calling ActorCritic.__init__
        nn.Module.__init__(self)
        
        # Store the required parameters
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        d_model_a, dim_ff_a = actor_hidden_dims
        d_model_c, dim_ff_c = critic_hidden_dims
        
        # Policy (Actor) - Now a Transformer
        self.actor = nn.ModuleDict({
            "embedding": nn.Linear(num_actor_obs, d_model_a),
            "transformer": TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=d_model_a,
                    nhead=nhead,
                    dim_feedforward=dim_ff_a,
                    activation=activation,
                    batch_first=True
                ),
                num_layers=num_layers
            ),
            "out": nn.Linear(d_model_a, num_actions)
        })

        self.critic = nn.ModuleDict({
            "embedding": nn.Linear(num_critic_obs, d_model_c),
            "transformer": TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=d_model_c,
                    nhead=nhead,
                    dim_feedforward=dim_ff_c,
                    activation=activation,
                    batch_first=True
                ),
                num_layers=num_layers
            ),
            "out": nn.Linear(d_model_c, 1)
        })

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        """Forward pass for inference.
        
        Args:
            observations: Input observations tensor
            
        Returns:
            actions: Predicted actions
        """
        return self.act_inference(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # observations: [batch, obs_dim]
        x = self.actor["embedding"](observations).unsqueeze(1)   # [B, 1, d_model]
        x = self.actor["transformer"](x)                         # [B, 1, d_model]
        mean = self.actor["out"](x.squeeze(1)) 
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        x = self.actor["embedding"](observations).unsqueeze(1)
        x = self.actor["transformer"](x)
        actions_mean = self.actor["out"](x.squeeze(1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        x = self.critic["embedding"](critic_observations).unsqueeze(1)
        x = self.critic["transformer"](x)
        value = self.critic["out"](x.squeeze(1))
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
