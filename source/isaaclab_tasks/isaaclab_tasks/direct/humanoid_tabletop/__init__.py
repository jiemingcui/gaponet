# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid Tabletop environment.
"""

import gymnasium as gym
import os
path = os.path.dirname(os.path.abspath(__file__))

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Tabletop-Direct-v0",
    entry_point=f"{__name__}.humanoid_tabletop_env:HumanoidTabletopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_tabletop_env_cfg:HumanoidTabletopEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(path, "robomimic/bc.json"),
    },
)
