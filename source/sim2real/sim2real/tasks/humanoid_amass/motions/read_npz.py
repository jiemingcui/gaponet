import os
import numpy as np

# 27 dofs
_dof_names = ['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
                'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
                'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
                'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
                'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
                'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
                'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
                'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
                'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
        

def print_all_data_name(data):
    # List all array names
    print("Array names:", data.files)

    # View content and shape of each array
    for name in data.files:
        print(f"\nArray name: {name}")
        print("Shape:", data[name].shape)
        print("Data type:", data[name].dtype)
    print(data["real_dof_positions_cmd"][0, 0])
    print(data["joint_sequence"])
    # import pdb; pdb.set_trace()
        

def process_all_amass_data(origin_path):
    npz_dir = origin_path 

    # Fields to process
    fields = [
        "real_dof_positions",
        "real_dof_positions_cmd",
        "real_dof_velocities",
        "real_dof_torques"
    ]

    # Use dict to store all concatenated results
    results = {f: [] for f in fields}

    for filename in os.listdir(npz_dir):
        npz_path = os.path.join(npz_dir, filename)
        data = np.load(npz_path)
        joint_sequence = data["joint_sequence"]  # 10 joint names

        # Calculate index of joint_sequence in _dof_names
        joint_indices = [ _dof_names.index(j) for j in joint_sequence ]
        
        for field in fields:
            arr = data[field]         # (10, 50)
            arr = arr.T               # (50, 10)
            new_arr = np.zeros((50, len(_dof_names)), dtype=arr.dtype)  # (50, 27)
            new_arr[:, joint_indices] = arr
            results[field].append(new_arr)

    # Concatenate data from all files
    for field in fields:
        # Result shape (n, 50, 27)
        results[field] = np.stack(results[field], axis=0)    # type:ignore

    # Final result example
    real_dof_positions_all = results["real_dof_positions"]  # (n, 50, 27)
    real_dof_positions_cmd_all = results["real_dof_positions_cmd"]
    real_dof_velocities_all = results["real_dof_velocities"]
    real_dof_torques_all = results["real_dof_torques"]

    np.savez(
        os.path.join(os.path.dirname(origin_path), "edited_27dof/merged_data.npz"),
        real_dof_positions=real_dof_positions_all,
        real_dof_positions_cmd=real_dof_positions_cmd_all,
        real_dof_velocities=real_dof_velocities_all,
        real_dof_torques=real_dof_torques_all,
        joint_sequence=joint_sequence
    )

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))


    npz_file_path = os.path.join(script_dir, './motion_amass/edited_27dof/motor_edited_extend_amass_merged_data.npz')
    npz_file_path = os.path.join(script_dir, './motion_amass/npz_zips/motor_edited_extend_amass_merged_data.npz')

    # Load .npz file
    data = np.load(npz_file_path)

    print_all_data_name(data)
    
    # process_all_amass_data(os.path.join(script_dir, 'motion_amass/origin'))
    
    