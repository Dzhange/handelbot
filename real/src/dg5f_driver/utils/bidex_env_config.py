from dataclasses import dataclass, field
from .arm_controller import ArmControllerConfig

@dataclass
class BiDexEnvConfig:
    enable_right_arm: bool = True
    enable_left_arm: bool = True
    enable_right_hand: bool = True
    enable_left_hand: bool = True
    
    right_arm_controller: ArmControllerConfig = field(default_factory=ArmControllerConfig)
    left_arm_controller: ArmControllerConfig = field(default_factory=ArmControllerConfig)
    
    randomize_init: int = 0
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)

    enable_arm_plot: bool = False
    enable_finger_plot: bool = False

    control_freq: int = 40
    policy_freq: int = 10
