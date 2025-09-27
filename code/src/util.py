import numpy as np

def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix (world <- body).

    Parameters
    ----------
    quat : np.ndarray
        Quaternion as [w, x, y, z]

    Returns
    -------
    R : np.ndarray
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize to avoid drift if quat is not unit length
    # If magnitude is 0, use identity (no rotation) matrix
    n = np.linalg.norm(quat)
    if n == 0:
        return np.eye(3)
    w, x, y, z = quat / n

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R
