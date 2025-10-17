import numpy as np

BASE_HOVER_THRUST = 0.26487

# K values tuned via visual inspection of oscillations (PID_k_test.py)
# https://www.youtube.com/watch?v=ZMI_kpNUgJM

# Proportional (Z altitude) and derivative (Z velocity) gain for thrust
KP_POS_Z = 1.0
KD_VEL_Z = 0.8

# Proportional (attitude) and derivative (angular velocity) gain for roll/pitch
KP_VEL = 0.06
KP_ATT_RP = 2.0
KD_ANG_RP = 3.8

# Proportional (yaw angle) and derivative (yaw rate) gain for yaw
KP_YAW = 1.0
KD_YAW = 3.0

class NeutralHoverController:
    def __init__(self, target_yaw: float = 0.0):
        """
        Neutral hover controller that damps motion while restoring level attitude and altitude.

        Parameters
        ----------
        target_yaw : float
            nominal heading (radians)
        """
        self.base_thrust = BASE_HOVER_THRUST
        self.target_yaw = target_yaw


    def update(self, pos, vel, rpy, ang_vel):
        """
        Parameters
        -----------
        pos: [x, y, z] position (m)
        vel: [vx, vy, vz] velocity (m/s)
        rpy: [roll, pitch, yaw] radian angles in euler
        ang_vel: [p, q, r] angular rates (rad/s)
        
        Returns
        -------
        dict of {thrust, roll, pitch, yaw} commands
        """

        # # Thrust (Z control): maintain neutral hover around z = 0.5
        # z = pos[2]
        # vz = vel[2]
        # thrust_cmd = self.base_thrust + KP_POS_Z * (-z + 0.5) - KD_VEL_Z * vz

        # Just using base hover for thrust (in RL context, let navigator agent go up/down)


        # Roll and Pitch based on current roll/pitch/yaw, angular velocity, and velocity
        roll, pitch, yaw = rpy
        p, q, r = ang_vel
        vx, vy = vel[0], vel[1]

        roll_target  = +KP_VEL * vy
        pitch_target = -KP_VEL * vx

        roll_cmd  = KP_ATT_RP * (roll - roll_target) + KD_ANG_RP * p
        pitch_cmd = KP_ATT_RP * (pitch - pitch_target) +  KD_ANG_RP * q

        # Yaw
        yaw_err = np.arctan2(np.sin(self.target_yaw - yaw), np.cos(self.target_yaw - yaw))
        yaw_cmd = -KP_YAW * yaw_err + KD_YAW * r


        # Clip commands
        # thrust_cmd = np.clip(thrust_cmd,  0.0, 0.35)
        roll_cmd  = np.clip(roll_cmd,  -1.0, 1.0)
        pitch_cmd = np.clip(pitch_cmd, -1.0, 1.0)
        yaw_cmd   = np.clip(yaw_cmd,   -1.0, 1.0)

        return {
            "thrust": self.base_thrust,
            "roll": roll_cmd,
            "pitch": pitch_cmd,
            "yaw": yaw_cmd,
        }
