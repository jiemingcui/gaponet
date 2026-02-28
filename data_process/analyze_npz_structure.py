"""
NPZ 数据分析脚本

此脚本用于：
1. 分析 merged_50Hz_31_10_payload.npz 文件结构
2. 对比配置文件中的维度假设与实际数据
3. 生成详细的对比报告

用法: python analyze_npz_structure.py
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('npz_analysis_report.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class NPZAnalyzer:
    """NPZ 文件分析器"""
    
    def __init__(self, npz_path: str):
        self.npz_path = Path(npz_path)
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.metadata: Dict[str, Any] = {}
        
    def load_file(self) -> bool:
        """加载 NPZ 文件"""
        try:
            if not self.npz_path.exists():
                logger.error(f"文件不存在: {self.npz_path}")
                return False
            
            self.data = np.load(self.npz_path, allow_pickle=True)
            logger.info(f"成功加载文件: {self.npz_path}")
            return True
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            return False
    
    def analyze_structure(self) -> Dict[str, Any]:
        """分析 NPZ 文件结构"""
        if self.data is None:
            logger.error("数据未加载")
            return {}
        
        analysis = {
            "file_path": str(self.npz_path),
            "file_size_mb": self.npz_path.stat().st_size / (1024 * 1024),
            "keys": list(self.data.files),
            "arrays": {}
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"NPZ 文件分析: {self.npz_path.name}")
        logger.info(f"{'='*60}")
        logger.info(f"文件大小: {analysis['file_size_mb']:.2f} MB")
        logger.info(f"数组数量: {len(self.data.files)}")
        logger.info(f"数组键: {self.data.files}")
        
        # 分析每个数组
        for key in self.data.files:
            arr = self.data[key]
            arr_info = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "ndim": arr.ndim,
                "size": arr.size,
                "nbytes": arr.nbytes
            }
            
            # 特殊处理 object 类型的数组
            if arr.dtype == object:
                if isinstance(arr, np.ndarray) and arr.ndim == 1:
                    arr_info["element_count"] = len(arr)
                    arr_info["element_types"] = []
                    arr_info["element_shapes"] = []
                    
                    # 分析每个元素
                    for i, elem in enumerate(arr[:5]):  # 只显示前5个
                        if isinstance(elem, np.ndarray):
                            arr_info["element_types"].append(f"ndarray-{elem.dtype}")
                            arr_info["element_shapes"].append(elem.shape)
                        elif isinstance(elem, (list, tuple)):
                            arr_info["element_types"].append(f"{type(elem).__name__}-len{len(elem)}")
                            arr_info["element_shapes"].append(None)
                        else:
                            arr_info["element_types"].append(type(elem).__name__)
                            arr_info["element_shapes"].append(None)
                    
                    if len(arr) > 5:
                        arr_info["element_types"].append("...")
                        arr_info["element_shapes"].append("...")
            
            analysis["arrays"][key] = arr_info
            
            # 打印详细信息
            logger.info(f"\n--- 数组: {key} ---")
            logger.info(f"  形状: {arr.shape}")
            logger.info(f"  数据类型: {arr.dtype}")
            logger.info(f"  维度: {arr.ndim}")
            logger.info(f"  元素数量: {arr.size}")
            logger.info(f"  内存大小: {arr.nbytes / 1024:.2f} KB")
            
            if arr.dtype == object:
                logger.info(f"  元素数量: {len(arr)}")
                if len(arr) > 0:
                    logger.info(f"  第一个元素类型: {type(arr[0])}")
                    if isinstance(arr[0], np.ndarray):
                        logger.info(f"  第一个元素形状: {arr[0].shape}")
        
        return analysis
    
    def analyze_joint_sequence(self) -> Dict[str, Any]:
        """分析关节序列"""
        if self.data is None or "joint_sequence" not in self.data:
            return {}
        
        joint_seq = self.data["joint_sequence"]
        result = {
            "key": "joint_sequence",
            "dtype": str(joint_seq.dtype),
            "length": len(joint_seq) if hasattr(joint_seq, '__len__') else 0
        }
        
        # 尝试转换为列表
        try:
            if joint_seq.dtype == object:
                joints = [str(j) for j in joint_seq]
            else:
                joints = joint_seq.tolist()
            
            result["joints"] = joints
            logger.info(f"\n--- 关节序列 ({len(joints)} 个关节) ---")
            for i, j in enumerate(joints):
                logger.info(f"  [{i}] {j}")
        except Exception as e:
            logger.warning(f"无法解析关节序列: {e}")
        
        return result
    
    def analyze_payloads(self) -> Dict[str, Any]:
        """分析负载信息"""
        if self.data is None or "payloads" not in self.data:
            return {}
        
        payloads = self.data["payloads"]
        result = {
            "key": "payloads",
            "shape": payloads.shape,
            "dtype": str(payloads.dtype),
            "unique_values": np.unique(payloads).tolist(),
            "value_counts": {}
        }
        
        # 统计每个 payload 的数量
        for p in np.unique(payloads):
            result["value_counts"][int(p)] = int(np.sum(payloads == p))
        
        logger.info(f"\n--- 负载信息 ---")
        logger.info(f"  形状: {payloads.shape}")
        logger.info(f"  唯一值: {result['unique_values']}")
        logger.info(f"  分布: {result['value_counts']}")
        
        return result
    
    def analyze_motion_data_dimensions(self) -> Dict[str, Any]:
        """分析运动数据的维度"""
        result = {}
        
        # 需要分析的数组
        data_keys = ["real_dof_positions", "real_dof_positions_cmd", 
                     "real_dof_velocities", "real_dof_torques"]
        
        for key in data_keys:
            if key not in self.data:
                continue
            
            arr = self.data[key]
            result[key] = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "sample_count": arr.shape[0] if arr.ndim >= 1 else 0,
                "dof_count": arr.shape[1] if arr.ndim >= 2 else 0
            }
            
            logger.info(f"\n--- {key} ---")
            logger.info(f"  形状: {arr.shape}")
            logger.info(f"  样本数量 (时间步): {result[key]['sample_count']}")
            logger.info(f"  DOF 数量: {result[key]['dof_count']}")
            
            # 分析每个样本的维度
            if arr.dtype == object and arr.ndim == 1:
                shapes = []
                for elem in arr[:3]:
                    if isinstance(elem, np.ndarray):
                        shapes.append(elem.shape)
                result[key]["sample_shapes"] = shapes
                logger.info(f"  前几个样本形状: {shapes}")
        
        return result


class ConfigDimensionVerifier:
    """配置文件维度验证器"""
    
    def __init__(self):
        self.config_assumptions: Dict[str, Any] = {}
        
    def load_rsl_rl_operator_cfg(self, cfg_path: str) -> Dict[str, Any]:
        """加载 rsl_rl_operator_cfg.py 中的维度假设"""
        assumptions = {
            "file": "rsl_rl_operator_cfg.py",
            "DeepONetActorCriticCfg": {}
        }
        
        # 从代码注释中提取的维度配置
        assumptions["DeepONetActorCriticCfg"] = {
            "branch_input_dims": "[20 * 62] = 1240",
            "trunk_input_dim": "32 (current_action 31 + payload 1)",
            "critic_input_dim": "1463",
            "model_input_dim": "31 + 93*4 = 403",
            "model_output_dim": "20 * 62 = 1240",
            "model_history_length": "4",
            "model_history_dim": "93"
        }
        
        self.config_assumptions["rsl_rl_operator_cfg"] = assumptions
        return assumptions
    
    def load_env_cfg_fourior(self, cfg_path: str) -> Dict[str, Any]:
        """加载 humanoid_operator_env_cfg_fourior.py 中的维度假设"""
        assumptions = {
            "file": "humanoid_operator_env_cfg_fourior.py",
            "env_config": {}
        }
        
        # 从配置文件和代码中提取的维度配置
        assumptions["env_config"] = {
            "action_space": "1 * 31 = 31 (所有 31 个关节都使用 delta action)",
            "num_sensor_positions": "20",
            "sensor_dim": "62 (joint_pos 31 + joint_vel*dt 31)",
            "model_history_length": "4",
            "model_history_dim": "93 (joint_pos 31 + joint_vel 31 + joint_target 31)",
            "delta_sensor_position": "True",
            "delta_sensor_value": "True"
        }
        
        self.config_assumptions["env_cfg_fourior"] = assumptions
        return assumptions
    
    def load_env_fourior(self, env_path: str) -> Dict[str, Any]:
        """加载 humanoid_operator_env_fourior.py 中的维度假设"""
        assumptions = {
            "file": "humanoid_operator_env_fourior.py",
            "env_class": {}
        }
        
        # 假设的维度（需要从代码中确认）
        assumptions["env_class"] = {
            "note": "需要从代码中进一步分析"
        }
        
        self.config_assumptions["env_fourior"] = assumptions
        return assumptions
    
    def verify_dimensions(self, npz_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """验证维度配置"""
        verification = {
            "matches": [],
            "mismatches": [],
            "warnings": []
        }
        
        # 获取 NPZ 数据中的关键维度
        motion_data = npz_analysis.get("motion_data_dims", {})
        joint_seq = npz_analysis.get("joint_sequence", {})
        
        # 1. 验证 DOF 数量
        if "real_dof_positions" in motion_data:
            npz_dof = motion_data["real_dof_positions"]["dof_count"]
            config_action = 31
            
            if npz_dof == config_action:
                verification["matches"].append(f"DOF 数量: NPZ={npz_dof}, 配置期望={config_action} ✓")
            else:
                verification["mismatches"].append(f"DOF 数量: NPZ={npz_dof}, 配置期望={config_action} ✗")
        
        # 2. 验证关节数量
        if "joint_sequence" in joint_seq:
            npz_joints = joint_seq.get("length", 0)
            # 配置中说是 31 DOF
            if npz_joints == 31:
                verification["matches"].append(f"关节数量: NPZ={npz_joints}, 配置期望=31 ✓")
            else:
                verification["mismatches"].append(f"关节数量: NPZ={npz_joints}, 配置期望=31 ✗")
        
        # 3. 验证传感器维度
        config_sensor_dim = 62  # joint_pos(31) + joint_vel*dt(31)
        config_sensor_positions = 20
        expected_sensor_total = config_sensor_positions * config_sensor_dim  # 1240
        
        verification["config_sensor_dim"] = f"{config_sensor_dim} 维/位置"
        verification["config_sensor_positions"] = f"{config_sensor_positions} 个位置"
        verification["expected_branch_input"] = f"{expected_sensor_total} = {config_sensor_positions} × {config_sensor_dim}"
        
        # 4. 验证 model history 维度
        config_model_history_dim = 93  # joint_pos(31) + joint_vel(31) + joint_target(31)
        config_model_history_length = 4
        expected_model_input = 31 + config_model_history_dim * config_model_history_length
        
        verification["config_model_history_dim"] = f"{config_model_history_dim} 维"
        verification["config_model_history_length"] = f"{config_model_history_length} 步"
        verification["expected_model_input"] = f"{expected_model_input} = 31 + {config_model_history_dim} × {config_model_history_length}"
        
        return verification
    
    def generate_report(self, npz_analysis: Dict, verification: Dict) -> str:
        """生成完整的对比报告"""
        report = []
        report.append("\n" + "="*80)
        report.append("NPZ 数据结构 vs 配置文件维度 对比报告")
        report.append("="*80)
        
        # 1. NPZ 数据概览
        report.append("\n【1. NPZ 文件数据概览】")
        report.append("-" * 40)
        if "arrays" in npz_analysis:
            for key, info in npz_analysis["arrays"].items():
                report.append(f"\n  {key}:")
                report.append(f"    形状: {info.get('shape', 'N/A')}")
                report.append(f"    数据类型: {info.get('dtype', 'N/A')}")
        
        # 2. 运动数据维度
        report.append("\n【2. 运动数据维度分析】")
        report.append("-" * 40)
        motion_dims = npz_analysis.get("motion_data_dims", {})
        for key, info in motion_dims.items():
            report.append(f"\n  {key}:")
            report.append(f"    样本数量: {info.get('sample_count', 'N/A')}")
            report.append(f"    DOF 数量: {info.get('dof_count', 'N/A')}")
        
        # 3. 关节序列
        report.append("\n【3. 关节序列分析】")
        report.append("-" * 40)
        joint_seq = npz_analysis.get("joint_sequence", {})
        if "joints" in joint_seq:
            report.append(f"  关节数量: {len(joint_seq['joints'])}")
            report.append("  关节列表:")
            for i, j in enumerate(joint_seq["joints"]):
                report.append(f"    [{i}] {j}")
        
        # 4. 负载信息
        report.append("\n【4. 负载信息】")
        report.append("-" * 40)
        payloads = npz_analysis.get("payloads", {})
        if "unique_values" in payloads:
            report.append(f"  唯一负载值: {payloads['unique_values']}")
            report.append(f"  负载分布: {payloads.get('value_counts', {})}")
        
        # 5. 配置文件的维度假设
        report.append("\n【5. 配置文件维度假设】")
        report.append("-" * 40)
        
        # rsl_rl_operator_cfg
        report.append("\n  5.1 rsl_rl_operator_cfg.py (DeepONetActorCriticCfg):")
        cfg = self.config_assumptions.get("rsl_rl_operator_cfg", {})
        deeponet = cfg.get("DeepONetActorCriticCfg", {})
        for k, v in deeponet.items():
            report.append(f"    {k}: {v}")
        
        # env_cfg_fourior
        report.append("\n  5.2 humanoid_operator_env_cfg_fourior.py:")
        env_cfg = self.config_assumptions.get("env_cfg_fourior", {})
        env_config = env_cfg.get("env_config", {})
        for k, v in env_config.items():
            report.append(f"    {k}: {v}")
        
        # 6. 验证结果
        report.append("\n【6. 维度验证结果】")
        report.append("-" * 40)
        
        if verification.get("matches"):
            report.append("\n  ✓ 匹配的维度:")
            for m in verification["matches"]:
                report.append(f"    {m}")
        
        if verification.get("mismatches"):
            report.append("\n  ✗ 不匹配的维度:")
            for m in verification["mismatches"]:
                report.append(f"    {m}")
        
        if verification.get("config_sensor_dim"):
            report.append("\n  传感器配置:")
            report.append(f"    每位置维度: {verification['config_sensor_dim']}")
            report.append(f"    位置数量: {verification['config_sensor_positions']}")
            report.append(f"    预期分支输入: {verification['expected_branch_input']}")
        
        if verification.get("config_model_history_dim"):
            report.append("\n  模型历史配置:")
            report.append(f"    历史维度: {verification['config_model_history_dim']}")
            report.append(f"    历史长度: {verification['config_model_history_length']}")
            report.append(f"    预期模型输入: {verification['expected_model_input']}")
        
        # 7. 总结
        report.append("\n【7. 总结】")
        report.append("-" * 40)
        match_count = len(verification.get("matches", []))
        mismatch_count = len(verification.get("mismatches", []))
        
        report.append(f"  匹配项: {match_count}")
        report.append(f"  不匹配项: {mismatch_count}")
        
        if mismatch_count == 0:
            report.append("\n  ✓ 所有维度验证通过！")
        else:
            report.append("\n  ✗ 存在维度不匹配，请检查配置文件")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


def main():
    """主函数"""
    logger.info("开始 NPZ 数据分析...")
    
    # NPZ 文件路径
    npz_path = "merged_50Hz_31_10_payload.npz"
    
    # 尝试多个可能的路径
    possible_paths = [
        npz_path,
        os.path.join(os.path.dirname(__file__), npz_path),
        os.path.join(os.path.dirname(__file__), "source", "sim2real", "sim2real", "tasks", 
                     "humanoid_operator", "motions", "motion_amass", "fourior", npz_path),
        r"c:\Users\12511\Desktop\gaponet\merged_50Hz_31_10_payload.npz",
    ]
    
    # 查找实际存在的文件
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            logger.info(f"找到文件: {p}")
            break
    
    if actual_path is None:
        logger.error("无法找到 NPZ 文件")
        # 列出目录中的 npz 文件
        logger.info("搜索目录中的 NPZ 文件...")
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith(".npz"):
                    logger.info(f"  找到: {os.path.join(root, f)}")
        return
    
    # 创建分析器
    analyzer = NPZAnalyzer(actual_path)
    
    # 加载文件
    if not analyzer.load_file():
        return
    
    # 分析结构
    analysis = analyzer.analyze_structure()
    
    # 分析关节序列
    joint_analysis = analyzer.analyze_joint_sequence()
    analysis["joint_sequence"] = joint_analysis
    
    # 分析负载
    payload_analysis = analyzer.analyze_payloads()
    analysis["payloads"] = payload_analysis
    
    # 分析运动数据维度
    motion_analysis = analyzer.analyze_motion_data_dimensions()
    analysis["motion_data_dims"] = motion_analysis
    
    # 创建配置验证器
    verifier = ConfigDimensionVerifier()
    
    # 加载配置假设
    verifier.load_rsl_rl_operator_cfg("rsl_rl_operator_cfg.py")
    verifier.load_env_cfg_fourior("humanoid_operator_env_cfg_fourior.py")
    verifier.load_env_fourior("humanoid_operator_env_fourior.py")
    
    # 验证维度
    verification = verifier.verify_dimensions(analysis)
    
    # 生成报告
    report = verifier.generate_report(analysis, verification)
    
    # 输出报告
    logger.info(report)
    
    # 保存报告到文件
    report_path = "npz_dimension_verification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"\n报告已保存到: {report_path}")
    
    # 打印最终结果
    logger.info("\n" + "="*60)
    logger.info("分析完成!")
    logger.info("="*60)
    
    return analysis, verification


if __name__ == "__main__":
    main()
