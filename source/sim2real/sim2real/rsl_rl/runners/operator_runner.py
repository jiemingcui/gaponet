# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from collections import deque

import time
import os
import statistics
import random

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from sim2real.rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import store_code_state
from sim2real.tasks.humanoid_operator.humanoid_operator_env import HumanoidOperatorEnv
from sim2real.rsl_rl.modules import *
from sim2real.rsl_rl.algorithms import *

class OperatorRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env: HumanoidOperatorEnv = env.unwrapped # type: ignore

        self.training_type = "rl"

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        obs = self.env.compute_operator_observation()
        num_obs = sum(o.size(1) for n, o in obs.items() if n != "critic")
        num_privileged_obs = obs["critic"].size(1)

        self.mismatch_model_input_dim = False
        model_obs_dim = self.env.compute_model_observation().shape[1]

        if self.policy_cfg['model_input_dim'] != model_obs_dim:
            self.mismatch_model_input_dim = True
        self.policy_cfg['model_input_dim'] = model_obs_dim
            
        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy = policy_class(
            0, 0, self.env.cfg.action_space, **self.policy_cfg
        ).to(self.device)

        # Runner handles the sensor model
        self.sensor_model = policy.model
        policy.model = None
        self.sensor_model_optimizer = torch.optim.Adam(self.sensor_model.parameters(), lr=self.alg_cfg["learning_rate"])

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        policy.model = self.sensor_model
        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.cfg.action_space],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self.replay_buffer_size = self.cfg["replay_buffer_size"]

        self.model_replay_buffer_size = self.cfg["model_replay_buffer_size"]
        self.model_learning_epochs = self.cfg["model_learning_epochs"]
        self.model_learning_steps = self.cfg["model_learning_steps"]
        self.model_sample_iterations = self.cfg["model_sample_iterations"]

        self.model_based_sensor = self.cfg["model_based_sensor"]
        self.num_steps_function = self.cfg["num_steps_function"]
        self.retrain_sensor_only = self.cfg["retrain_sensor_only"]
        if self.model_based_sensor:
            assert self.num_steps_per_env % self.num_steps_function == 0
        else:
            assert self.num_steps_function == self.num_steps_per_env
        self.model_learning_interval = self.cfg["model_learning_interval"]

        self.randomize_dynamics = self.cfg["randomize_dynamics"]
        self.direct_sample_envs = self.cfg["direct_sample_envs"]
        self.full_trajectory_sampling = self.cfg["full_trajectory_sampling"]

        self.eval_after_training = self.cfg["eval_after_training"]

    def sample_functions_and_sensors(self, replay_buffer):
        samples = zip(*random.sample(replay_buffer, k=self.env.num_sensor_positions))
        function_coords, sub_env_sensor_data, motion_coords = samples
        cat_function_coords = {}
        cat_motion_coords = []
        for item in function_coords:
            for name, value in item.items():
                if name not in cat_function_coords:
                    cat_function_coords[name] = []
                cat_function_coords[name].append(value)
        for name, value in cat_function_coords.items():
            cat_function_coords[name] = torch.cat(value, dim=0)
        for item in zip(*motion_coords):
            cat_motion_coords.append(torch.cat(item, dim=0))
        return cat_function_coords, torch.cat(sub_env_sensor_data, dim=0), tuple(cat_motion_coords)
    
    def sample_model_pairs(self, replay_buffer, batch_size):
        samples = zip(*random.sample(replay_buffer, k=batch_size))
        model_inputs, model_outputs = samples
        return torch.cat(model_inputs, dim=0), torch.cat(model_outputs, dim=0)
    
    def learn_sensor_model(self):
        replay_buffer = []

        self.sensor_model.train()
        for epoch in range(self.model_learning_epochs):
            with torch.inference_mode():
                for _ in range(self.model_sample_iterations):
                    model_pairs = self.env.compute_model_pairs(add_noise=True)
                    model_inputs = model_pairs["obs"].to(self.device)
                    model_outputs = model_pairs["sensor"].flatten(1, 2).to(self.device)
                    replay_buffer.append((model_inputs, model_outputs))
            
                if len(replay_buffer) > self.model_replay_buffer_size:
                    replay_buffer = replay_buffer[-self.model_replay_buffer_size:]

                model_inputs, model_outputs = self.sample_model_pairs(replay_buffer, len(replay_buffer))
                assert model_inputs.shape[0] % self.model_learning_steps == 0

            learning_batch_size = model_inputs.shape[0] // self.model_learning_steps
            mean_loss = 0

            for i in range(self.model_learning_steps):
                with torch.inference_mode():
                    inputs, outputs = model_inputs[i*learning_batch_size:(i+1)*learning_batch_size], model_outputs[i*learning_batch_size:(i+1)*learning_batch_size]
                sensor = self.sensor_model(inputs.clone())
                loss = ((sensor - outputs) ** 2).mean(dim=0).sum()
                self.sensor_model_optimizer.zero_grad()
                loss.backward()
                self.sensor_model_optimizer.step()
                mean_loss += loss.item()
            
            print(f"Sensor model learning: Epoch {epoch+1} loss: {mean_loss / self.model_learning_steps}")

        self.env.reset_all_dynamics()

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        if self.retrain_sensor_only:
            try:
                resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            except:
                del loaded_dict["model_state_dict"]["model.0.weight"]
                resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=False)
            return loaded_dict["infos"]
        else:
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training and not self.mismatch_model_input_dim:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"]) # type: ignore
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def save(self, *args, **kwargs):
        if self.sensor_model is not None:
            self.alg.policy.model = self.sensor_model
        super().save(*args, **kwargs)
        if self.sensor_model is not None:
            self.alg.policy.model = None # type: ignore

    def learn(self, num_learning_iterations: int, **kwargs):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter # type: ignore

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        self.train_mode()  # switch to train mode (for dropout for example)
        if self.retrain_sensor_only:
            self.learn_sensor_model()
            if self.log_dir is not None and not self.disable_logs:
                self.save(os.path.join(self.log_dir, f"model_sensor_retrained.pt"))
            return
        
        # if self.model_based_sensor:
        #     self.learn_sensor_model()
        #     self.sensor_model.eval()

        # if hasattr(self.alg.policy, "input_normalizer") and self.empirical_normalization:
        #     self.alg.policy.input_normalizer = self.obs_normalizer

        # Book keeping
        ep_infos = []
        replay_buffer = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        num_sub_envs = self.env.num_sub_environments
        num_sensor_positions = self.env.num_sensor_positions
        num_steps_per_function = self.num_steps_per_env // self.num_steps_function

        if not self.model_based_sensor:
            with torch.inference_mode():
                while len(replay_buffer) < num_sensor_positions:
                    for _ in range(num_sensor_positions):
                        print(f"Filling replay buffer: {len(replay_buffer)+1} / {num_sensor_positions}")
                        # sample sensors
                        function_coords, sub_env_sensor_data, motion_coords = self.env.step_sensor(resample=True)
                        replay_buffer.append((function_coords, sub_env_sensor_data, motion_coords))

        for it in range(start_iter, tot_iter):
            start = time.time()
            if (it - start_iter) % self.model_learning_interval == 0 and self.model_based_sensor:
                self.learn_sensor_model()
                self.sensor_model.eval()

            # Rollout
            if not self.model_based_sensor or not self.direct_sample_envs:
                with torch.inference_mode():
                    for _ in range(num_sensor_positions):
                        # sample sensors
                        function_coords, sub_env_sensor_data, motion_coords = self.env.step_sensor(resample=True, min_available_length=num_steps_per_function)
                        replay_buffer.append((function_coords, sub_env_sensor_data, motion_coords))
                    
                    if len(replay_buffer) > self.replay_buffer_size:
                        replay_buffer = replay_buffer[-self.replay_buffer_size:]

            with torch.inference_mode():
                for _ in range(self.num_steps_function):
                    if self.randomize_dynamics and self.model_based_sensor:
                        self.env.sample_all_dynamics(sample_payload=not self.full_trajectory_sampling)

                    if not self.model_based_sensor or not self.direct_sample_envs:
                        function_coords, sensor_data, motion_coords = self.sample_functions_and_sensors(replay_buffer)
                        self.env.create_function(function_coords, sensor_data, motion_coords)
                    elif not self.full_trajectory_sampling:
                        self.env.sample_all_environments(min_available_length=num_steps_per_function)
                    
                    if self.model_based_sensor:
                        sensor_data = self.sensor_model(self.env.compute_model_observation().to(self.device)).reshape(self.env.num_envs, self.env.num_sensor_positions, -1)
                        self.env.set_sensor_data(sensor_data.to(self.env.device))
                    
                    if not 'obs' in locals() or not self.full_trajectory_sampling:
                        obs = self.env.compute_operator_observation()
                        privileged_obs = obs["critic"].to(self.device)
                        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
                        obs = torch.cat([v for k, v in obs.items() if k != "critic"], dim=1).to(self.device)
                    
                    for i in range(num_steps_per_function):
                        # Sample actions
                        actions = self.alg.act(obs, privileged_obs)
                        # Step the environment
                        obs, rewards, dones, infos = self.env.step_operator(actions.to(self.env.device), # type: ignore
                                                                            motion_coords if not self.model_based_sensor else None) 
                        if self.model_based_sensor:
                            sensor_data = self.sensor_model(self.env.compute_model_observation().to(self.device)).reshape(self.env.num_envs, self.env.num_sensor_positions, -1)
                            self.env.set_sensor_data(sensor_data.to(self.env.device))
                        observation = self.env.compute_operator_observation()

                        # Move to device
                        obs, privileged_obs, rewards, dones = (
                            torch.cat([v for k, v in observation.items() if k != "critic"], dim=1).to(self.device), 
                            observation["critic"].to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device)
                        )

                        if not self.full_trajectory_sampling and i == num_steps_per_function - 1:
                            dones |= True

                        # perform normalization
                        # obs = self.obs_normalizer(obs)
                        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
                        
                        # process the step
                        self.alg.process_env_step(rewards, dones, infos)

                        # book keeping
                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            # Update rewards
                            cur_reward_sum += rewards / num_steps_per_function

                            # Update episode length
                            cur_episode_length += 1
                            # Clear data for completed episodes
                            # -- common
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs) # type: ignore

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path) # type: ignore

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.eval_after_training:
            self.eval()

    @torch.inference_mode()
    def eval(self):
        self.alg.policy.model = self.sensor_model
        self.alg.policy.eval()

        self.env.mode = 'play'
        self.env.cfg.mode = 'play'
        self.env.add_noise = False
        self.env.init_play_environments(reload_motion=True)

        policy: DeepONetActorCritic = self.alg.policy # type: ignore
        inf_policy = policy.full_forward

        obs = self.env._get_observations() # type: ignore
        i = 0
        while True:
            print(f'Inference Step {i}')
            actions = inf_policy(obs) # type: ignore
            obs, _, _, _, _ = self.env.step(actions)
            i += 1

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)