import os
import numpy as np
import matplotlib.pyplot as plt

def plot_joint_positions(data, save_path, motion_frequency, motion_num, joint_sequence):
    """
    data:  shape: (time_length, num_envs==1, num_data_type, 10)
    """

    has_sim_dof_positions = False
    if data.shape[2] == 3:
        robot_dof_positions = data[:, 0, 0][2: ]   # shape: (time_length, 10)
        dof_target_pos = data[:, 0, 1][2: ]        # shape: (time_length, 10)
        real_dof_positions = data[:, 0, 2][2: ]    # shape: (time_length, 10)
    else:
        robot_dof_positions = data[:, 0, 0][2: ]   # shape: (time_length, 10)
        sim_dof_positions = data[:, 0, 1][2: ]    # shape: (time_length, 10)
        dof_target_pos = data[:, 0, 2][2: ]        # shape: (time_length, 10)
        real_dof_positions = data[:, 0, 3][2: ]    # shape: (time_length, 10)
        has_sim_dof_positions = True

    fig, axs = plt.subplots(3, 5, figsize=(60, 24))  # 3 rows, 5 columns, 15 subplots total
    axs = axs.flatten()  # flatten for 1D indexing

    steps = range(0, robot_dof_positions.shape[0])
    for i in range(len(joint_sequence)):
        axs[i].plot(steps, robot_dof_positions[:, i], label='delta action dof positions')
        axs[i].plot(steps, real_dof_positions[:, i], label='real dof positions')
        axs[i].plot(steps, dof_target_pos[:, i], label='target dof positions')
        if has_sim_dof_positions:
            axs[i].plot(steps, sim_dof_positions[:, i], label='sim dof positions')
        axs[i].set_title(f'{joint_sequence[i]}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend(fontsize=8)
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint Positions & Target', fontsize=18)
    plt.savefig(os.path.join(save_path, 'joint_position.pdf'), dpi=300)


def plot_joint_velocity(data, save_path, motion_frequency, motion_num, joint_sequence):
    robot_dof_velocities = data[:, 0, 1][2: 50]   # shape: (time_length, 10)
    real_dof_velocities = data[:, 0, 4][2: 50]

    fig, axs = plt.subplots(2, 5, figsize=(18, 8))  # 2 rows, 5 columns, 10 subplots total
    axs = axs.flatten()  # flatten for 1D indexing

    steps = range(0, robot_dof_velocities.shape[0])
    for i in range(10):
        axs[i].plot(steps, robot_dof_velocities[:, i], label='delta action dof velocity')
        axs[i].plot(steps, real_dof_velocities[:, i], label='real dof velocity')
        axs[i].set_title(f'{joint_sequence[i]}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend(fontsize=8)
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(save_path, 'joint_velocity.png'))

def plot_joint_torque(data, save_path, motion_frequency, motion_num, joint_sequence):
    robot_dof_torques = data[:, 0, 5][2: 50]
    real_dof_torques = data[:, 0, 6][2: 50]

    tau_external = data[:, 0, 7][2: 50]

    fig, axs = plt.subplots(2, 5, figsize=(18, 8))  # 2 rows, 5 columns, 10 subplots total
    axs = axs.flatten()  # flatten for 1D indexing

    steps = range(0, robot_dof_torques.shape[0])
    for i in range(10):
        
        axs[i].plot(steps, robot_dof_torques[:, i], label='delta action dof torque')
        axs[i].plot(steps, real_dof_torques[:, i], label='real dof torque')
        axs[i].plot(steps, tau_external[:, i], label='tau_external')
        axs[i].set_title(f'{joint_sequence[i]}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend(fontsize=8)
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(save_path, 'joint_torque.png'))

def plot_joint_acc(data, save_path, motion_frequency, motion_num, joint_sequence):
    dof_acc = data[:, 0, 8][2: 50]

    fig, axs = plt.subplots(2, 5, figsize=(18, 8))  # 2 rows, 5 columns, 10 subplots total
    axs = axs.flatten()  # flatten for 1D indexing
    
    steps = range(0, dof_acc.shape[0])
    for i in range(10):
        axs[i].plot(steps, dof_acc[:, i], label='accleration dof')
        
        axs[i].set_title(f'{joint_sequence[i]}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend(fontsize=8)
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(save_path, 'joint_acc.png'))