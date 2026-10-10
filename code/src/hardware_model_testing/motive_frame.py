"""Convert Motive Y-up poses into the Z-up frame used by the trained policy."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


# Motive (x, y-up, z) -> policy (x, y, z-up): (x, y, z)_m -> (x, z, y)_p
_P = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def motive_to_policy_position(
    xyz_motive: np.ndarray,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Map Motive metres (Y-up) to policy metres (Z-up)."""
    v = _P @ np.asarray(xyz_motive, dtype=np.float64).reshape(3)
    if flip_x:
        v[0] *= -1.0
    if flip_y:
        v[1] *= -1.0
    return v.astype(np.float32)


def motive_to_policy_quaternion(
    quat_xyzw_motive: np.ndarray,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Map a Motive quaternion (xyzw) into the policy world frame."""
    r_m = R.from_quat(np.asarray(quat_xyzw_motive, dtype=np.float64).reshape(4))
    f = np.diag([(-1.0 if flip_x else 1.0), (-1.0 if flip_y else 1.0), 1.0])
    r_p = f @ _P @ r_m.as_matrix() @ _P @ f
    q = R.from_matrix(r_p).as_quat()
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)
