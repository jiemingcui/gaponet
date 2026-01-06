import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from torch.distributions import Normal
from sim2real.tasks.humanoid_operator.humanoid_operator_env import HumanoidOperatorEnv
from sim2real.rsl_rl.networks.multi_res_branch_net import MultiResolutionBranchNet
from rsl_rl.modules.actor_critic import ActorCritic

class TrunkNet(nn.Module):
    """Trunk network of DeepONet architecture"""
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepONetActorCritic(ActorCritic):
    """DeepONet-based Actor-Critic network for PPO with multi-resolution branch network"""
    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        action_dim: int,
        model_history_length: int,
        model_history_dim: int,
        branch_input_dims: List[int],  # List of input dimensions for different resolutions
        trunk_input_dim: int,
        critic_input_dim: int,
        model_input_dim: int,
        model_output_dim: int,
        branch_hidden_dim: int,
        trunk_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        model_hidden_dims: list[int],
        model_pretrained_path: str,
        activation: str = "elu"
    ):
        super(ActorCritic, self).__init__()
        self.is_recurrent = False

        self.action_dim = action_dim
        self.model_history_length = model_history_length
        self.model_history_dim = model_history_dim
        
        # Initialize networks
        self.branch_net = MultiResolutionBranchNet(
            input_dims=branch_input_dims,
            hidden_dim=branch_hidden_dim,
            output_dim=action_dim * 16,
            activation=activation
        )
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_hidden_dims, action_dim * 16)

        self.branch_input_dims = branch_input_dims
        self.critic_input_dim = critic_input_dim
        self.total_branch_dim = sum(branch_input_dims)
        self.total_trunk_dim = trunk_input_dim
        
        # Actor output layer
        # self.actor_output = nn.Sequential(
        #     nn.Linear(128, action_dim),
        #     nn.Tanh()  # Bound actions to [-1, 1]
        # )
        self._input_normalizer = None
        
        # Critic network
        critic_layers = []
        prev_dim = critic_input_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.model_input_dim = model_input_dim
        self.model_hidden_dims = model_hidden_dims
        self.model_output_dim = model_output_dim
        self.init_sensor_model()
        if model_pretrained_path != "":
            loaded_dict = torch.load(model_pretrained_path, map_location=self.model[0].weight.device)
            state_dict = loaded_dict["model_state_dict"]
            model_dict = {k: v for k, v in state_dict.items() if "model" in k}
            self.load_state_dict(model_dict, strict=False)

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 2.995) # keep initial std close to 0.05

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        self._init_weights()

    def _init_weights(self):
        self.trunk_net.net[-1].weight.data.uniform_(-0.003, 0.003)
        self.trunk_net.net[-1].bias.data.uniform_(-0.003, 0.003)

    def init_sensor_model(self):
        model_layers = []
        prev_dim = self.model_input_dim
        for hidden_dim in self.model_hidden_dims:
            model_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        model_layers.append(nn.Linear(prev_dim, self.model_output_dim))
        self.model = nn.Sequential(*model_layers)

    def chunk_flattened_inputs(self, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Chunk flattened inputs into a list of tensors
        """
        start_idx = 0
        branch_inputs = []
        for branch_input_dim in self.branch_input_dims:
            branch_inputs.append(inputs[:, start_idx:start_idx+branch_input_dim])
            start_idx += branch_input_dim
        
        trunk_input = inputs[:, start_idx:]
        return branch_inputs, trunk_input

    def forward(self, branch_inputs: List[torch.Tensor], trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            branch_inputs: List of inputs for different resolutions of branch network
            trunk_input: Input for trunk network
            
        Returns:
            Tuple of (actions, value)
        """
        # Process through DeepONet
        batch_size = branch_inputs[0].shape[0]
        branch_out = self.branch_net(branch_inputs)
        trunk_out = self.trunk_net(trunk_input)
        combined = branch_out * trunk_out
        
        # Get actions and value
        # actions = self.actor_output(combined)
        actions = combined.view(batch_size, -1, self.action_dim).sum(dim=1)
        return actions
    
    def update_distribution(self, inputs: torch.Tensor):
        """
        Update the action distribution
        """
        branch_inputs, trunk_input = self.chunk_flattened_inputs(inputs)
        mean = self.forward(branch_inputs, trunk_input)
        self.distribution = Normal(mean, torch.exp(self.log_std))
    
    def act_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get actions for inference
        """
        branch_inputs, trunk_input = self.chunk_flattened_inputs(inputs)
        actions = self.forward(branch_inputs, trunk_input)
        return actions
    
    def evaluate(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Evaluate actions and compute value estimates
        
        Args:
            inputs: Input for trunk network
            
        Returns:
            value
        """
        # Get value estimate
        value = self.critic(inputs)
        return value
    
    def model_sensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get sensor output
        """
        sensors = self.model(inputs)
        return sensors
    
    def full_forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass through the network
        """
        if self.training:
            self.eval()
            print("WARNING: full_forward is called during evaluation only")
            
        model_obs = obs["model"]
        operator_obs = obs["operator"]
        sensors = self.model_sensor(model_obs)

        branch_input = operator_obs["branch"] # type: ignore
        branch_input[:, :sensors.shape[1]] = sensors
        trunk_input = operator_obs["trunk"] # type: ignore
        if self._input_normalizer is not None:
            inputs = torch.cat([branch_input, trunk_input], dim=1)
            inputs = self._input_normalizer(inputs)
            actions = self.forward(*self.chunk_flattened_inputs(inputs))
        else:
            actions = self.forward([branch_input], trunk_input)

        return actions
    
    def reset_model_history(self, env_ids: torch.Tensor | None = None):
        if 'model_history' in self.__dict__:
            if env_ids is None:
                self.model_history[:] *= 0.
                return
            self.model_history[env_ids] *= 0.
    
    def apply_delta_action(self, env: HumanoidOperatorEnv, action: torch.Tensor):
        assert hasattr(env, 'robot'), "env must have a robot"
        assert hasattr(env, 'delta_action_joint_indices'), "env must have a delta_action_joint_indices"
        
        if 'model_history' not in self.__dict__:
            self.model_history = torch.zeros(env.num_envs, self.model_history_length, self.model_history_dim, device=env.device)
        delta_action_joint_indices = env.delta_action_joint_indices

        joint_pos = env.robot.data.joint_pos[:, delta_action_joint_indices]
        joint_vel = env.robot.data.joint_vel[:, delta_action_joint_indices]
        joint_obs = torch.cat([joint_pos, joint_vel], dim=1)
        model_obs = torch.cat([joint_obs, self.model_history.flatten(1, 2)], dim=1)

        self.model_history = self.model_history.roll(1, dims=1)
        self.model_history[:, 0, :] = joint_obs.clone()

        branch_inputs = self.model_sensor(model_obs)
        if action.size(1) != self.action_dim:
            input_action = action[:, delta_action_joint_indices]
            assert input_action.size(1) == self.action_dim, "action size must match action dim"
        else:
            input_action = action

        trunk_input = input_action
        delta_actions = self.forward([branch_inputs], trunk_input)

        full_action = input_action + delta_actions
        if input_action.size(1) != action.size(1):
            action[:, delta_action_joint_indices] = full_action
        else:
            action = full_action

        return action

        

        
        
        
        
        
