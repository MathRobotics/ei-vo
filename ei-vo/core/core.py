# franka_core/schema.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict

@dataclass
class Trajectory:
    t:  np.ndarray          # (T,)
    q:  np.ndarray          # (T, 7)
    dq: Optional[np.ndarray] = None
    ddq:Optional[np.ndarray] = None
    meta: Dict = None

# franka_core/model.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class RobotModel:
    name: str
    joint_names: List[str]          # ex. panda0_joint1..7
    limits: Optional[np.ndarray] = None  # (7,2)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        if self.limits is None: return q
        out = q.copy()
        for i,(lo,hi) in enumerate(self.limits):
            if lo < hi: out[:,i] = np.clip(out[:,i], lo, hi)
        return out
