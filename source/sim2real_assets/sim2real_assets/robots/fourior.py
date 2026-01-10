import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

assets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fourior_with_hand_fix_usd_filename = './usds/fourior/gr3v2_2_2.usd'
fourior_with_hand_fix_usd_file_path = os.path.join(assets_dir, fourior_with_hand_fix_usd_filename)

FOURIOR_WITH_HAND_FIX_URDF_PATH = os.path.join(assets_dir, "urdfs/fourior/gr3v2_2_2.urdf")

FOURIOR_CFG_WITH_HAND_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=fourior_with_hand_fix_usd_file_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,

            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # ".*_hip_yaw_joint": 0.0,
            # ".*_hip_roll_joint": 0.0,
            # ".*_hip_pitch_joint": 0.0,  # -16 degrees
            # ".*_knee_joint": 0.0,  # 45 degrees
            # ".*_ankle_pitch_joint": 0.0,  # -30 degrees
            # ".*_ankle_roll_joint": 0.0,  # -30 degrees
            # "torso_joint": 0.0,
            # ".*_shoulder_pitch_joint": 0.0,
            # ".*_shoulder_roll_joint": 0.0,
            # ".*_shoulder_yaw_joint": 0.0,
            # ".*_elbow_joint": 0.0,
            # ".*_wrist_roll_joint": 0.0,
            # ".*_wrist_pitch_joint": 0.0,
            # ".*_wrist_yaw_joint": 0.0,

            ".*_joint": 0.0,
        },
        joint_vel={".*_joint": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0
            },
            damping={
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 4.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                # ".*_wrist_pitch_joint",
                # ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=30,
            velocity_limit_sim=20,
            stiffness={
                ".*_shoulder_pitch_joint": 200.0,
                ".*_shoulder_roll_joint": 200.0,
                ".*_shoulder_yaw_joint": 200.0,
                ".*_elbow_joint": 100.0,
                ".*_wrist_roll_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 10.0,
                ".*_wrist_roll_joint": 5.0,
            },
        ),
        
    },
)


fourior_with_hand_fix_payload_usd_filename = './usds/fourior_payload/gr3v2_2_2_payload.usd'
fourior_with_hand_fix_payload_usd_file_path = os.path.join(assets_dir, fourior_with_hand_fix_payload_usd_filename)

FOURIOR_WITH_HAND_FIX_PAYLOAD_URDF_PATH = os.path.join(assets_dir, "urdfs/fourior_payload/gr3v2_2_2_payload.urdf")

FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=fourior_with_hand_fix_payload_usd_file_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,

            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # ".*_hip_yaw_joint": 0.0,
            # ".*_hip_roll_joint": 0.0,
            # ".*_hip_pitch_joint": 0.0,  # -16 degrees
            # ".*_knee_pitch_joint": 0.0,  # 45 degrees
            # ".*_ankle_pitch_joint": 0.0,  # -30 degrees
            # ".*_ankle_roll_joint": 0.0,  # -30 degrees
            # "waist_.*_joint": 0.0,
            # ".*_shoulder_pitch_joint": 0.0,
            # ".*_shoulder_roll_joint": 0.0,
            # ".*_shoulder_yaw_joint": 0.0,
            # ".*_elbow_pitch_joint": 0.0,
            # ".*_wrist_roll_joint": 0.0,
            # ".*_wrist_pitch_joint": 0.0,
            # ".*_wrist_yaw_joint": 0.0,

            ".*_joint": 0.0,
        },
        joint_vel={".*_joint": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
                "waist_.*_joint"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
                "waist_.*_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0
            },
            damping={
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 4.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=30,
            velocity_limit_sim=20,
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 100.0,
                ".*_elbow_pitch_joint": 100.0,
                ".*_wrist_yaw_joint": 50.0,
                ".*_wrist_pitch_joint": 50.0,
                ".*_wrist_roll_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_roll_joint": 1.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_.*_joint"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                "head_.*_joint": 50.0,
            },
            damping={
                "head_.*_joint": 5.0,
            },
        ),
        
    },
)
