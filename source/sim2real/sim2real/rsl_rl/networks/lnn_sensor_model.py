import torch
import torch.nn as nn
from typing import List

class LNNSensorModel(nn.Module):
    def __init__(
        self,
        model_input_dim: int,
        model_output_dim: int,
        model_hidden_dims: List[int],
        model_state_dim: int,
        time_constant: float = 1.0,
        step_dt: float = 0.02
    ):
        super(LNNSensorModel, self).__init__()

        self.model_input_dim = model_input_dim
        self.model_output_dim = model_output_dim
        self.model_hidden_dims = model_hidden_dims
        self.model_state_dim = model_state_dim

        model_layers = []
        prev_dim = self.model_input_dim + self.model_state_dim
        for hidden_dim in self.model_hidden_dims:
            model_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        model_layers.append(nn.Linear(prev_dim, self.model_state_dim))
        self.model = nn.Sequential(*model_layers)

        self.proj = nn.Linear(self.model_state_dim, self.model_output_dim)
        self.bias = nn.Parameter(torch.zeros(self.model_state_dim))
        self.initial_hidden_state = nn.Parameter(torch.zeros(self.model_state_dim))
        self.hidden_state = None

        self.time_constant = time_constant
        self.step_dt = step_dt


    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor | None = None,
                time_constant: float | None = None,
                step_dt: float | None = None):
        if time_constant is None:
            time_constant = self.time_constant
        if step_dt is None:
            step_dt = self.step_dt

        if hidden_state is None:
            hidden_state = self.initial_hidden_state.clone()
            
        fx = self.model(torch.cat([x, hidden_state], dim=1))
        hidden_state = (hidden_state + step_dt * fx) / (1 + step_dt * (1 / time_constant + fx))

        return self.proj(hidden_state), hidden_state
    
    def reset_hidden_state(self, hidden_state: torch.Tensor | None = None):
        self.hidden_state = hidden_state

    @torch.no_grad()
    def inference(self, x: torch.Tensor,
                time_constant: float | None = None,
                step_dt: float | None = None):
        if time_constant is None:
            time_constant = self.time_constant
        if step_dt is None:
            step_dt = self.step_dt

        if self.hidden_state is None:
            self.hidden_state = self.initial_hidden_state.clone()

        output, hidden_state = self.forward(x, self.hidden_state, time_constant, step_dt)
        self.hidden_state = hidden_state
        return output
        