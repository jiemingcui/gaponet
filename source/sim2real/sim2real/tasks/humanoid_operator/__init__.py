# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Operator-Delta-Action",
    entry_point=f"{__name__}.humanoid_operator_env:HumanoidOperatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_operator_env_cfg:HumanoidOperatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorRunnerCfg",
    },
)

gym.register(
    id="Isaac-Humanoid-Operator-Delta-Action-Fourior",
    entry_point=f"{__name__}.humanoid_operator_env_fourior:HumanoidOperatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_operator_env_cfg_fourior:HumanoidOperatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorFourierRunnerCfg",
    },
)

gym.register(
    id="Isaac-Humanoid-Operator-Delta-Action-Vanilla",
    entry_point=f"{__name__}.humanoid_operator_env:HumanoidOperatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_operator_env_cfg:HumanoidOperatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorVanillaRunnerCfg",
    },
)

