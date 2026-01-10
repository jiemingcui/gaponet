import torch
from typing import List, Tuple, Dict
from isaaclab.assets import Articulation, RigidObject

def get_sensor_positions(dof_names: List[str], sensor_positions: List[Dict[str, float]]) -> torch.Tensor:
    sensor_positions_list = []
    for sensor_position in sensor_positions:
        sensor_tensor = torch.zeros(len(dof_names))
        for name, value in sensor_position.items():
            sensor_tensor[dof_names.index(name)] = value
        sensor_positions_list.append(sensor_tensor)
    return torch.stack(sensor_positions_list, dim=0)

def reset_masses(asset: Articulation | RigidObject, env_ids: torch.Tensor):
    env_ids = env_ids.cpu()
    masses = asset.data.default_mass[env_ids]
    set_masses(asset, masses, env_ids)

def set_masses(asset: Articulation | RigidObject, masses: torch.Tensor, env_ids: torch.Tensor, recompute_inertia: bool = True):
    # resolve environment ids
    env_ids = env_ids.cpu()
    masses = masses.cpu()

    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    
    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)