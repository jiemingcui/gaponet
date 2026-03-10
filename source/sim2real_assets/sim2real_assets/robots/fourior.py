"""
Fourior 机器人配置文件

本文件定义了 Fourior 机器人在 Isaac Lab 仿真环境中的配置，包括：
1. 无负载版本机器人配置 (FOURIOR_CFG_WITH_HAND_FIX)
2. 带负载版本机器人配置 (FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD)

每个配置包含：
- 机器人资产路径（USD/URDF文件）
- 刚体物理属性
- 关节初始状态
- 执行器参数（刚度、阻尼、力矩限制等）
"""

# 导入 Isaac Lab 仿真工具模块
import isaaclab.sim as sim_utils
# 导入执行器配置类型（虽然定义了但未全部使用）
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
# 导入关节系统配置类
from isaaclab.assets.articulation import ArticulationCfg
import os

# 获取资产根目录路径（当前文件的上两级目录）
# 例如：如果当前文件在 robots/ 目录下，则 assets_dir 指向 sim2real_assets/ 目录
assets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 无负载版本机器人配置
# ============================================================================

# 无负载版本 USD 文件相对路径
fourior_with_hand_fix_usd_filename = './usds/fourior/gr3v2_2_2.usd'
# 无负载版本 USD 文件完整路径
fourior_with_hand_fix_usd_file_path = os.path.join(assets_dir, fourior_with_hand_fix_usd_filename)

# 无负载版本 URDF 文件完整路径（用于参考，实际使用 USD 文件）
FOURIOR_WITH_HAND_FIX_URDF_PATH = os.path.join(assets_dir, "urdfs/fourior/gr3v2_2_2.urdf")

# 无负载版本机器人配置
FOURIOR_CFG_WITH_HAND_FIX = ArticulationCfg(
    # 机器人生成配置：从 USD 文件加载机器人模型
    spawn=sim_utils.UsdFileCfg(
        # USD 文件路径（Isaac Sim 使用的 3D 场景文件格式）
        usd_path=fourior_with_hand_fix_usd_file_path,
        # 是否激活接触传感器（用于检测碰撞和接触力）
        activate_contact_sensors=True,
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 是否禁用重力（False 表示启用重力）
            disable_gravity=False,
            # 是否保留加速度信息（False 表示不保留，节省计算资源）
            retain_accelerations=False,
            # 线性阻尼系数（0.0 表示无阻尼，用于模拟空气阻力等）
            linear_damping=0.0,
            # 角速度阻尼系数（0.0 表示无阻尼）
            angular_damping=0.0,
            # 最大线性速度限制（单位：m/s，防止数值不稳定）
            max_linear_velocity=1000.0,
            # 最大角速度限制（单位：rad/s，防止数值不稳定）
            max_angular_velocity=1000.0,
            # 最大去穿透速度（单位：m/s，用于解决碰撞穿透问题）
            max_depenetration_velocity=1.0,
        ),
        # 关节系统根节点属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 是否启用自碰撞检测（False 表示不检测，避免机器人部件之间相互碰撞）
            enabled_self_collisions=False,
            # 位置求解器迭代次数（影响碰撞检测精度，值越大越精确但计算越慢）
            solver_position_iteration_count=4,
            # 速度求解器迭代次数（影响物理模拟稳定性）
            solver_velocity_iteration_count=4,
            # 是否固定根链接（True 表示机器人根部固定，不会移动）
            fix_root_link=True,
        ),
    ),
    # 机器人初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        # 机器人根部的初始位置 (x, y, z)，单位：米
        # (0.0, 0.0, 1.0) 表示在原点上方 1 米处
        pos=(0.0, 0.0, 1.0),
        # 关节初始位置配置（使用正则表达式匹配关节名称）
        # 所有匹配的关节初始位置设为 0.0 弧度
        # 以下是被注释掉的详细关节配置（可根据需要取消注释并设置特定值）
        joint_pos={
            # ".*_hip_yaw_joint": 0.0,        # 髋关节偏航角
            # ".*_hip_roll_joint": 0.0,       # 髋关节横滚角
            # ".*_hip_pitch_joint": 0.0,      # 髋关节俯仰角（建议值：-16度）
            # ".*_knee_joint": 0.0,           # 膝关节（建议值：45度）
            # ".*_ankle_pitch_joint": 0.0,    # 踝关节俯仰角（建议值：-30度）
            # ".*_ankle_roll_joint": 0.0,     # 踝关节横滚角（建议值：-30度）
            # "torso_joint": 0.0,             # 躯干关节
            # ".*_shoulder_pitch_joint": 0.0, # 肩关节俯仰角
            # ".*_shoulder_roll_joint": 0.0,  # 肩关节横滚角
            # ".*_shoulder_yaw_joint": 0.0,   # 肩关节偏航角
            # ".*_elbow_joint": 0.0,          # 肘关节
            # ".*_wrist_roll_joint": 0.0,     # 腕关节横滚角
            # ".*_wrist_pitch_joint": 0.0,    # 腕关节俯仰角
            # ".*_wrist_yaw_joint": 0.0,      # 腕关节偏航角

            # 使用正则表达式匹配所有关节，统一设置为 0.0 弧度
            ".*_joint": 0.0,
        },
        # 关节初始速度配置（所有关节初始速度设为 0.0 rad/s）
        joint_vel={".*_joint": 0.0},
    ),
    # 软关节位置限制因子（0.9 表示关节位置限制为硬限制的 90%，提供安全裕度）
    soft_joint_pos_limit_factor=0.9,
    # 执行器配置：定义各关节组的控制参数
    # ImplicitActuatorCfg 使用隐式 PD 控制器（位置-速度控制）
    actuators={
        # 腿部执行器组：包括髋关节、膝关节和躯干关节
        "legs": ImplicitActuatorCfg(
            # 使用正则表达式匹配的关节名称列表
            joint_names_expr=[
                ".*_hip_yaw_joint",      # 髋关节偏航角（左右旋转）
                ".*_hip_roll_joint",     # 髋关节横滚角（内外旋转）
                ".*_hip_pitch_joint",    # 髋关节俯仰角（前后摆动）
                ".*_knee_joint",         # 膝关节
                "torso_joint"            # 躯干关节
            ],
            # 力矩限制（单位：N·m），防止执行器输出过大
            effort_limit_sim=300,
            # 速度限制（单位：rad/s），防止关节运动过快
            velocity_limit_sim=100.0,
            # 刚度系数（单位：N·m/rad），控制位置跟踪的响应速度
            # 值越大，位置跟踪越精确但可能更不稳定
            stiffness={
                ".*_hip_yaw_joint": 150.0,    # 髋关节偏航：中等刚度
                ".*_hip_roll_joint": 150.0,   # 髋关节横滚：中等刚度
                ".*_hip_pitch_joint": 200.0,  # 髋关节俯仰：较高刚度（主要承重关节）
                ".*_knee_joint": 200.0,       # 膝关节：较高刚度（主要承重关节）
                "torso_joint": 200.0,         # 躯干关节：较高刚度
            },
            # 阻尼系数（单位：N·m·s/rad），控制速度响应，减少振荡
            # 值越大，系统越稳定但响应可能变慢
            damping={
                ".*_hip_yaw_joint": 5.0,      # 髋关节偏航：中等阻尼
                ".*_hip_roll_joint": 5.0,     # 髋关节横滚：中等阻尼
                ".*_hip_pitch_joint": 5.0,    # 髋关节俯仰：中等阻尼
                ".*_knee_joint": 5.0,         # 膝关节：中等阻尼
                "torso_joint": 5.0,           # 躯干关节：中等阻尼
            },
        ),
        # 脚部执行器组：包括踝关节（俯仰和横滚）
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            # 脚部关节力矩限制较小（因为主要起平衡作用）
            effort_limit_sim=100,
            velocity_limit_sim=100.0,
            # 脚部关节刚度较低（允许一定柔顺性，适应地面不平）
            stiffness={
                ".*_ankle_pitch_joint": 20.0,  # 踝关节俯仰：低刚度
                ".*_ankle_roll_joint": 20.0    # 踝关节横滚：低刚度
            },
            # 脚部关节阻尼适中（保持稳定但不过度限制）
            damping={
                ".*_ankle_pitch_joint": 4.0,   # 踝关节俯仰：中等阻尼
                ".*_ankle_roll_joint": 4.0     # 踝关节横滚：中等阻尼
            },
        ),
        # 手臂执行器组：包括肩关节、肘关节和腕关节
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",  # 肩关节俯仰角
                ".*_shoulder_roll_joint",   # 肩关节横滚角
                ".*_shoulder_yaw_joint",    # 肩关节偏航角
                ".*_elbow_joint",           # 肘关节
                ".*_wrist_roll_joint",      # 腕关节横滚角
                # 以下两个腕关节在当前配置中被注释掉（未启用）
                # ".*_wrist_pitch_joint",   # 腕关节俯仰角
                # ".*_wrist_yaw_joint",     # 腕关节偏航角
            ],
            # 手臂关节力矩限制较小（手臂通常不需要很大力矩）
            effort_limit_sim=30,
            # 手臂关节速度限制较低（精细操作需要较慢速度）
            velocity_limit_sim=20,
            # 手臂关节刚度配置（从肩到腕逐渐降低）
            stiffness={
                ".*_shoulder_pitch_joint": 200.0,  # 肩关节俯仰：高刚度
                ".*_shoulder_roll_joint": 200.0,   # 肩关节横滚：高刚度
                ".*_shoulder_yaw_joint": 200.0,    # 肩关节偏航：高刚度
                ".*_elbow_joint": 100.0,           # 肘关节：中等刚度
                ".*_wrist_roll_joint": 50.0,       # 腕关节横滚：低刚度（精细操作）
            },
            # 手臂关节阻尼配置（从肩到腕逐渐降低）
            damping={
                ".*_shoulder_pitch_joint": 20.0,   # 肩关节俯仰：高阻尼
                ".*_shoulder_roll_joint": 20.0,    # 肩关节横滚：高阻尼
                ".*_shoulder_yaw_joint": 20.0,     # 肩关节偏航：高阻尼
                ".*_elbow_joint": 10.0,            # 肘关节：中等阻尼
                ".*_wrist_roll_joint": 5.0,        # 腕关节横滚：低阻尼
            },
        ),
        
    },
)


# ============================================================================
# 带负载版本机器人配置
# ============================================================================
# 注意：带负载版本与无负载版本的主要区别：
# 1. 启用了自碰撞检测（enabled_self_collisions=True）
# 2. 关节命名略有不同（如 knee_joint vs knee_pitch_joint）
# 3. 手臂执行器包含更多腕关节（wrist_yaw, wrist_pitch, wrist_roll）
# 4. 增加了头部执行器配置

# 带负载版本 USD 文件相对路径（使用独立的 payload 版本 USD）
fourior_with_hand_fix_payload_usd_filename = './usds/fourior_payload/gr3v2_2_2_payload.usd'
# 带负载版本 USD 文件完整路径
fourior_with_hand_fix_payload_usd_file_path = os.path.join(assets_dir, fourior_with_hand_fix_payload_usd_filename)

# 带负载版本 URDF 文件完整路径（payload 版本 URDF）
FOURIOR_WITH_HAND_FIX_PAYLOAD_URDF_PATH = os.path.join(assets_dir, "urdfs/fourior_payload/gr3v2_2_2_payload.urdf")

# 带负载版本机器人配置
FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD = ArticulationCfg(
    # 机器人生成配置：从 USD 文件加载带负载的机器人模型
    spawn=sim_utils.UsdFileCfg(
        # 带负载版本 USD 文件路径
        usd_path=fourior_with_hand_fix_payload_usd_file_path,
        # 激活接触传感器（用于检测负载与环境的接触）
        activate_contact_sensors=True,
        # 刚体物理属性配置（与无负载版本相同）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # 关节系统根节点属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 启用自碰撞检测（True 表示检测，带负载时可能需要避免负载与身体碰撞）
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            # 固定根链接（机器人根部固定）
            fix_root_link=True,
        ),
    ),
    # 机器人初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        # 机器人根部的初始位置 (x, y, z)，单位：米
        pos=(0.0, 0.0, 1.0),
        # 关节初始位置配置（使用正则表达式匹配关节名称）
        # 注意：带负载版本的关节命名略有不同（如 knee_pitch_joint vs knee_joint）
        joint_pos={
            # ".*_hip_yaw_joint": 0.0,        # 髋关节偏航角
            # ".*_hip_roll_joint": 0.0,       # 髋关节横滚角
            # ".*_hip_pitch_joint": 0.0,      # 髋关节俯仰角（建议值：-16度）
            # ".*_knee_pitch_joint": 0.0,     # 膝关节俯仰角（注意：命名不同，建议值：45度）
            # ".*_ankle_pitch_joint": 0.0,    # 踝关节俯仰角（建议值：-30度）
            # ".*_ankle_roll_joint": 0.0,     # 踝关节横滚角（建议值：-30度）
            # "waist_.*_joint": 0.0,          # 腰部关节（注意：命名不同）
            # ".*_shoulder_pitch_joint": 0.0, # 肩关节俯仰角
            # ".*_shoulder_roll_joint": 0.0,   # 肩关节横滚角
            # ".*_shoulder_yaw_joint": 0.0,    # 肩关节偏航角
            # ".*_elbow_pitch_joint": 0.0,     # 肘关节俯仰角（注意：命名不同）
            # ".*_wrist_roll_joint": 0.0,      # 腕关节横滚角
            # ".*_wrist_pitch_joint": 0.0,     # 腕关节俯仰角
            # ".*_wrist_yaw_joint": 0.0,       # 腕关节偏航角

            # 使用正则表达式匹配所有关节，统一设置为 0.0 弧度
            ".*_joint": 0.0,
        },
        # 关节初始速度配置（所有关节初始速度设为 0.0 rad/s）
        joint_vel={".*_joint": 0.0},
    ),
    # 软关节位置限制因子（0.9 表示关节位置限制为硬限制的 90%）
    soft_joint_pos_limit_factor=0.9,
    # 执行器配置：定义各关节组的控制参数
    actuators={
        # 腿部执行器组：包括髋关节、膝关节和腰部关节
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",      # 髋关节偏航角
                ".*_hip_roll_joint",     # 髋关节横滚角
                ".*_hip_pitch_joint",    # 髋关节俯仰角
                ".*_knee_pitch_joint",   # 膝关节俯仰角（注意：命名与无负载版本不同）
                "waist_.*_joint"         # 腰部关节（注意：命名与无负载版本的 torso_joint 不同）
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            # 刚度配置（与无负载版本相同）
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
                "waist_.*_joint": 200.0,
            },
            # 阻尼配置（与无负载版本相同）
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
                "waist_.*_joint": 5.0,
            },
        ),
        # 脚部执行器组：包括踝关节（俯仰和横滚）
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=100.0,
            # 脚部关节刚度较低（与无负载版本相同）
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0
            },
            damping={
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 4.0,
            },
        ),
        # 手臂执行器组：包括肩关节、肘关节和完整的腕关节（三个自由度）
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",  # 肩关节俯仰角
                ".*_shoulder_roll_joint",   # 肩关节横滚角
                ".*_shoulder_yaw_joint",    # 肩关节偏航角
                ".*_elbow_pitch_joint",     # 肘关节俯仰角（注意：命名不同）
                # 带负载版本包含完整的腕关节三个自由度（无负载版本只有 wrist_roll）
                ".*_wrist_yaw_joint",       # 腕关节偏航角
                ".*_wrist_pitch_joint",     # 腕关节俯仰角
                ".*_wrist_roll_joint",      # 腕关节横滚角
            ],
            effort_limit_sim=30,
            velocity_limit_sim=20,
            # 手臂关节刚度配置（注意：带负载版本的刚度值较低，可能为了适应负载）
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,  # 肩关节俯仰：中等刚度（比无负载版本低）
                ".*_shoulder_roll_joint": 100.0,   # 肩关节横滚：中等刚度
                ".*_shoulder_yaw_joint": 100.0,    # 肩关节偏航：中等刚度
                ".*_elbow_pitch_joint": 100.0,     # 肘关节：中等刚度
                ".*_wrist_yaw_joint": 50.0,        # 腕关节偏航：低刚度
                ".*_wrist_pitch_joint": 50.0,      # 腕关节俯仰：低刚度
                ".*_wrist_roll_joint": 50.0,       # 腕关节横滚：低刚度
            },
            # 手臂关节阻尼配置（注意：带负载版本的阻尼值较低）
            damping={
                ".*_shoulder_pitch_joint": 2.0,    # 肩关节俯仰：低阻尼（比无负载版本低）
                ".*_shoulder_roll_joint": 2.0,     # 肩关节横滚：低阻尼
                ".*_shoulder_yaw_joint": 2.0,      # 肩关节偏航：低阻尼
                ".*_elbow_pitch_joint": 2.0,       # 肘关节：低阻尼
                ".*_wrist_yaw_joint": 1.0,         # 腕关节偏航：极低阻尼
                ".*_wrist_pitch_joint": 1.0,       # 腕关节俯仰：极低阻尼
                ".*_wrist_roll_joint": 1.0,        # 腕关节横滚：极低阻尼
            },
        ),
        # 头部执行器组：带负载版本特有的配置
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_.*_joint"  # 头部关节（用于头部运动控制）
            ],
            # 头部关节力矩限制较大（可能需要支撑头部重量）
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            # 头部关节刚度中等（平衡稳定性和灵活性）
            stiffness={
                "head_.*_joint": 50.0,
            },
            # 头部关节阻尼中等
            damping={
                "head_.*_joint": 5.0,
            },
        ),
        
    },
)

    