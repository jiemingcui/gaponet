import os
import numpy as np

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
    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 查看每个数组的内容和形状
    for name in data.files:
        print(f"\n数组名称: {name}")
        print("形状:", data[name].shape)
        print("数据类型:", data[name].dtype)
    print(data["real_dof_positions_cmd"][0][1])

def delete_data(data, new_file_path):
    # 新建一个字典存放处理后的数据
    new_data = {}

    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 查看每个数组的内容和形状，并处理
    for name in data.files:
        print(f"\n数组名称: {name}")
        arr = data[name]
        # 兼容list或ndarray
        arr = np.array(arr)
        print("原始形状:", arr.shape)
        print("原始数据类型:", arr.dtype)

        arr_new = arr[:-3]
        print("处理后形状:", arr_new.shape)
        new_data[name] = arr_new

    np.savez(new_file_path, **new_data)
    print(f"\n处理后的数据已保存至: {new_file_path}")

def process_and_save_npz(data, output_file: str, dof_names: list[str]):
    """
    读取 .npz 文件，将数组扩展到 (27, 49, 23) 并保存为新文件。

    Args:
        input_file (str): 输入的 .npz 文件路径。
        output_file (str): 输出的 .npz 文件路径。
        dof_names (list[str]): 所有关节的名称列表 (_dof_names)。
    """
    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 提取最后一个数组 (joint sequence)
    joint_sequence = data[data.files[-1]]  # 假设最后一个数组是 joint_sequence
    print("\nJoint Sequence:", joint_sequence)

    # 获取 joint_sequence 中每个 joint 在 _dof_names 中的索引
    joint_indices = []
    for joint_name in joint_sequence:
        if joint_name in dof_names:
            joint_indices.append(dof_names.index(joint_name))
        else:
            raise ValueError(f"Joint name '{joint_name}' not found in _dof_names.")

    print("\nJoint indices in _dof_names:", joint_indices)

    # 扩展所有前面的数组到 (27, 49, len(_dof_names))
    expanded_data = {}
    dof_count = len(dof_names)

    for name in data.files[:-1]:  # 遍历除 joint_sequence 以外的数组
        array = data[name]
        print(f"\n处理数组: {name}, 原始形状: {array.shape}")

        # 检查数组形状是否为 (27, 49)
        if array.shape != (27, 49):
            raise ValueError(f"Array '{name}' does not have the expected shape (27, 49).")

        # 初始化扩展后的数组
        expanded_array = np.zeros((27, 49, dof_count), dtype=array.dtype)

        # 按索引填充值
        for i, joint_index in enumerate(joint_indices):
            expanded_array[i, :, joint_index] = array[i, :]

        # 保存扩展后的数组
        expanded_data[name] = expanded_array
        print(f"{name} 已扩展到形状: {expanded_array.shape}")

    # 将 joint_sequence 也保存
    expanded_data["joint_sequence"] = joint_sequence

    # 保存为新的 npz 文件
    np.savez(output_file, **expanded_data)
    print(f"\n扩展后的数据已保存到: {output_file}")


if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    npz_file_path = os.path.join(script_dir, './humanoid_dance.npz')
    npz_file_path = os.path.join(script_dir, './motor.npz')
    npz_file_path = os.path.join(script_dir, './motor_edited.npz')
    npz_file_path = os.path.join(script_dir, './motor_edited_extend.npz')

    # 加载 .npz 文件
    data = np.load(npz_file_path)

    print_all_data_name(data)
    # delete_data(data, os.path.join(script_dir, 'motor_edited.npz'))
    # process_and_save_npz(data, os.path.join(script_dir, "./motor_edited_extend.npz"), _dof_names)


"""
数组名称: fps
形状: ()
数据类型: int64

数组名称: dof_names
形状: (28,)
数据类型: <U16

数组名称: body_names
形状: (15,)
数据类型: <U15

数组名称: dof_positions
形状: (902, 28)
数据类型: float32

数组名称: dof_velocities
形状: (902, 28)
数据类型: float32

数组名称: body_positions
形状: (902, 15, 3)
数据类型: float32

数组名称: body_rotations
形状: (902, 15, 4)
数据类型: float32

数组名称: body_linear_velocities
形状: (902, 15, 3)
数据类型: float32

数组名称: body_angular_velocities
形状: (902, 15, 3)
数据类型: float32
"""