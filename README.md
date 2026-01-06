# Training
Train MLP Delta Action Model
```./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Humanoid-Motor-Direct-v0 \
--num_envs=128 --max_iterations 100000 --experiment_name Sim2Real --run_name delta_action_mlp \
--device cuda 
```

## trunk 10dim
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real --letter amass --run_name delta_action_mlp --device cuda env.mode=train --headless
```

## trunk 11dim
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real --letter amass --run_name delta_action_mlp_payload --device cuda env.mode=train --headless
```

## Fourior
```

## trunk 11dim
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action-Fourior --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real --letter amass --run_name delta_action_fourior_payload --device cuda env.mode=train --headless
```
```

```
python scripts/rsl_rl/play.py  --task Isaac-Humanoid-Operator-Delta-Action   --checkpoint model_17950.pt --num_envs 20 --headless
```

# Transformer training
```
python scripts/rsl_rl/train_transformer.py --task Isaac-Humanoid-Amass-Delta-Action-Transformer --num_envs=4096 --max_iterations 100000 --experiment_name Sim2Real --letter amass --run_name delta_action_transformer_only --device cuda env.mode=train --headless
```

```
python3 scripts/rsl_rl/play.py --task Isaac-Humanoid-Amass-Delta-Action-Transformer --checkpoint model_17950.pt --num_envs 20 --headless --load_run 2025-09-16_18-01-32_delta_action_transformer_only
```

# MLP training
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Amass-Delta-Action-MLP --num_envs=4096 --max_iterations 100000 --experiment_name Sim2Real --letter amass --run_name delta_action_mlp_only --device cuda env.mode=train --headless
```
```
python scripts/rsl_rl/play.py  --task Isaac-Humanoid-Amass-Delta-Action-MLP   --checkpoint model_17950.pt --num_envs 20 --headless
```

350
