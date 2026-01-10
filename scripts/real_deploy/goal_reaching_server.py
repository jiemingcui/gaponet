import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import torch
import numpy as np
from pynput import keyboard

H1_2_NUM_MOTOR = 27

H1_2_JOINT_LIMITS = np.array([
    [-0.43, 0.43],      # 0  left_hip_yaw
    [-3.14, 2.5],       # 1  left_hip_pitch
    [-0.43, 3.14],      # 2  left_hip_roll
    [-0.26, 2.05],      # 3  left_knee
    [-0.897334, 0.523598], # 4  left_ankle_pitch
    [-0.261799, 0.261799], # 5  left_ankle_roll
    [-0.43, 0.43],      # 6  right_hip_yaw
    [-3.14, 2.5],       # 7  right_hip_pitch
    [-3.14, 0.43],      # 8  right_hip_roll
    [-0.26, 2.05],      # 9  right_knee
    [-0.897334, 0.523598], # 10 right_ankle_pitch
    [-0.261799, 0.261799], # 11 right_ankle_roll
    [-3.14, 1.57],      # 12 torso
    [-3.14, 1.57],      # 13 left_shoulder_pitch
    [-0.38, 3.4],       # 14 left_shoulder_roll
    [-3.01, 2.66],      # 15 left_shoulder_yaw
    [-2.53, 1.6],       # 16 left_elbow_pitch
    [-2.967, 2.967],    # 17 left_elbow_roll
    [-0.471, 0.349],    # 18 left_wrist_pitch
    [-1.012, 1.012],    # 19 left_wrist_yaw
    [-1.57, 3.14],      # 20 right_shoulder_pitch
    [-3.4, 0.38],       # 21 right_shoulder_roll
    [-2.66, 3.01],      # 22 right_shoulder_yaw
    [-1.6, 2.53],       # 23 right_elbow_pitch
    [-2.967, 2.967],    # 24 right_elbow_roll
    [-0.471, 0.349],    # 25 right_wrist_pitch
    [-1.012, 1.012]     # 26 right_wrist_yaw
])

class H1_2_JointIndex:
    # legs
    LeftHipYaw = 0
    LeftHipPitch = 1
    LeftHipRoll = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipYaw = 6
    RightHipPitch = 7
    RightHipRoll = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    # torso
    WaistYaw = 12
    # arms
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    LeftWristPitch = 18
    LeftWristYaw = 19
    RightShoulderPitch = 20
    RightShoulderRoll = 21
    RightShoulderYaw = 22
    RightElbow = 23
    RightWristRoll = 24
    RightWristPitch = 25
    RightWristYaw = 26

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class H1_2_GoalReachingServer:
    num_key_payloads = 2
    action_joint_indices = [13, 15, 14, 16, 20, 22, 21, 23, 17, 24, 18, 25, 19, 26]
    policy_path = "logs/exported/policy(5).pt"

    class obs_scales:
        joint_pos = 1.0
        joint_vel = 0.05

    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02  # [20ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

        self.started = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.joint_action = np.zeros(H1_2_NUM_MOTOR, dtype=np.float32)
        self.soft_joint_limit = H1_2_JOINT_LIMITS * 0.99
        self.soft_joint_limit[[18, 25]] *= 100.0

        self.joint_pos = np.zeros(H1_2_NUM_MOTOR, dtype=np.float32)
        self.joint_vel = np.zeros(H1_2_NUM_MOTOR, dtype=np.float32)
        self.goal_command = np.zeros((self.num_key_payloads, 3), dtype=np.float32)
        self.payload_masses = np.zeros(self.num_key_payloads, dtype=np.float32) + 0.001
        self.last_action = np.zeros(len(self.action_joint_indices), dtype=np.float32)

        self.current_key_payload = 0

    @torch.inference_mode()
    def GetAction(self):
        self.goal_command[:] = np.clip(self.goal_command, -1.0, 1.0)
        
        observation = torch.from_numpy(np.concatenate([
            self.joint_pos[self.action_joint_indices] * self.obs_scales.joint_pos,
            self.joint_vel[self.action_joint_indices] * self.obs_scales.joint_vel,
            self.goal_command.flatten(),
            self.payload_masses,
            self.last_action
        ])).float()
        observation = observation.unsqueeze(0).to(self.device)

        action = self.policy(observation)
        action = torch.clamp(action, -40., 40.)
        action = action.squeeze(0).cpu().numpy()

        # action = np.clip(action, -0.5, 0.5)
        print(action)

        self.last_action[:] = action
        self.joint_action[self.action_joint_indices] = action

    def Init(self):

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 1)

        # load policy #
        self.policy = torch.jit.load(self.policy_path, map_location="cpu").to(self.device)
        self.policy.eval()

        # keyboard #
        self.keyboard_listener = keyboard.Listener(on_press=self.KeyboardHandler)
        self.keyboard_listener.start()

    def KeyboardHandler(self, x: keyboard.Key | keyboard.KeyCode | None):
        if x is None:
            return
        if isinstance(x, keyboard.KeyCode):
            key_name = x.char
        else:
            key_name = x.name
        
        if key_name.isdigit():
            self.current_key_payload = np.clip(int(key_name), 0, self.num_key_payloads - 1) # type: ignore
            print(f"Current key payload: {self.current_key_payload}")
        elif key_name == '+':
            self.payload_masses[self.current_key_payload] += 0.01
            print(f"Payload {self.current_key_payload} mass: {self.payload_masses[self.current_key_payload]}")
        elif key_name == '-':
            self.payload_masses[self.current_key_payload] -= 0.01
            print(f"Payload {self.current_key_payload} mass: {self.payload_masses[self.current_key_payload]}")
        elif key_name == 'up':
            self.goal_command[self.current_key_payload, 2] += 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")
        elif key_name == 'down':
            self.goal_command[self.current_key_payload, 2] -= 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")
        elif key_name == 'left':
            self.goal_command[self.current_key_payload, 0] -= 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")
        elif key_name == 'right':
            self.goal_command[self.current_key_payload, 0] += 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")
        elif key_name == ',':
            self.goal_command[self.current_key_payload, 1] += 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")
        elif key_name == '.':
            self.goal_command[self.current_key_payload, 1] -= 0.01
            print(f"Goal command {self.current_key_payload}: {self.goal_command[self.current_key_payload]}")

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)
        
        self.start_joint_pos = self.joint_pos.copy()
        self.start_time = time.monotonic()
        self.first_step = True
        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

        input("Initialization complete. Press Enter to start.")
        self.started = True

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

        self.joint_pos[:] = np.array([motor.q for motor in self.low_state.motor_state])[:27].astype(np.float32)
        self.joint_vel[:] = np.array([motor.dq for motor in self.low_state.motor_state])[:27].astype(np.float32)

        joint_limit_exceeded = False
        if np.any(self.joint_pos < self.soft_joint_limit[:, 0]):
            print("Joint position below soft joint limit:", np.arange(H1_2_NUM_MOTOR)[self.joint_pos < self.soft_joint_limit[:, 0]])
            joint_limit_exceeded = True
        if np.any(self.joint_pos > self.soft_joint_limit[:, 1]):
            print("Joint position above soft joint limit:", np.arange(H1_2_NUM_MOTOR)[self.joint_pos > self.soft_joint_limit[:, 1]])
            joint_limit_exceeded = True
        
        if joint_limit_exceeded:
            print("Emergency stop due to joint limit exceeded")
            self.joint_action[:] = self.joint_pos
            for i in range(50):
                self.PublishLowCmd()
                time.sleep(0.02)
            exit()

    def LowCmdWrite(self):
        if not self.started:
            alpha = np.clip((time.monotonic() - self.start_time) / self.duration_, 0.0, 1.0)
            self.joint_action[:] = self.start_joint_pos * (1 - alpha)
        else:
            self.GetAction()
            # if not self.first_step:
            #     self.GetAction()
            # else:
            #     if not 'first_action' in self.__dict__:
            #         self.GetAction()
            #         self.first_action = self.joint_action.copy()
            #         self.start_joint_pos[:] = 0.
            #         self.start_time = time.monotonic()

            #     if time.monotonic() - self.start_time > self.duration_:
            #         self.first_step = False
            #     else:
            #         alpha = np.clip((time.monotonic() - self.start_time) / self.duration_, 0.0, 1.0)
            #         self.joint_action[:] = self.start_joint_pos * (1 - alpha) + self.first_action * alpha
        self.PublishLowCmd()

    def PublishLowCmd(self):
        self.time_ += self.control_dt_
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(H1_2_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode =  1 # type: ignore # 1:Enable, 0:Disable
            self.low_cmd.motor_cmd[i].tau = 0.0  # type: ignore
            self.low_cmd.motor_cmd[i].q = float(self.joint_action[i])  # type: ignore
            self.low_cmd.motor_cmd[i].dq = 0.0  # type: ignore
            self.low_cmd.motor_cmd[i].kp = 100.0 if i not in [17, 18, 19, 24, 25, 26] else 50.0  # type: ignore
            self.low_cmd.motor_cmd[i].kd = 2.0 if i not in [17, 18, 19, 24, 25, 26] else 1.0  # type: ignore

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    h1_2_goal_reaching_server = H1_2_GoalReachingServer()
    h1_2_goal_reaching_server.Init()
    h1_2_goal_reaching_server.Start()

    while True:
        time.sleep(1)