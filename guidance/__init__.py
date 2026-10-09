"""测试阶段可选的导引律模块。

这里的模块不参与训练、经验回放或策略更新；它们只在评估脚本中生成
候选动作，然后仍交由环境中的 COLREGs 过滤器和 3DOF 动力学执行。
"""

from .proportional_navigation import (
    PNConfig,
    PNDiagnostics,
    PNVesselParameters,
    ProportionalNavigationGuidance,
)

__all__ = [
    "PNConfig",
    "PNDiagnostics",
    "PNVesselParameters",
    "ProportionalNavigationGuidance",
]
