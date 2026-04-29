import numpy as np

import casadi as ca

from nmpc_path_follower.MPC_minimal import MPCMinimal


class RigidBody2DEuler(MPCMinimal):
    def __init__(self, N=15, dt=0.1):
        super().__init__(N, dt, solver_type="ipopt")
    
    @property
    def nx(self):
        return 3
    
    @property
    def nu(self):
        return 2
    
    @property
    def Q(self):
        return ca.diag([10.0, 10.0, 0.01])
    
    @property
    def Qn(self):
        return ca.diag([10.0, 10.0, 0.01])
    
    @property
    def Q_pos(self):
        return ca.diag([10.0, 10.0])
    
    @property
    def Q_yaw(self):
        return ca.diag([0.01])
    
    @property
    def Qn_pos(self):
        return ca.diag([10.0, 10.0])
    
    @property
    def Qn_yaw(self):
        return ca.diag([0.01])
    
    @property
    def R(self):
        return ca.diag([1.0, 1.0])
    
    def dynamics(self, x, u):
        psi, v, w = x[2], u[0], u[1]
        return ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            w
        )

    def stage_cost(self, x, u, x_ref=None, u_ref=None):
        if x_ref is None:
            x_ref = ca.zeros(self.nx)
        if u_ref is None:
            u_ref = ca.zeros(self.nu)
            
        # dx = x - x_ref
        # du = u - u_ref
        # return ca.mtimes([dx.T, self.Q, dx]) + ca.mtimes([du.T, self.R, du])
        
        _p_r = x[0:2]
        _p_r_sp = x_ref[0:2]
        _q_r = x[2]
        _q_r_sp = x_ref[2]
        
        dp = _p_r - _p_r_sp
        # https://www.reddit.com/r/gamedev/comments/1onzya/what_is_the_best_way_to_calculate_angle/#:~:text=given:%20x%20=%20(%20a%20%2D,is%20right%20of%20another%20angle.
        dq = ca.atan2(ca.sin(_q_r - _q_r_sp),ca.cos(_q_r - _q_r_sp))
        
        return ca.mtimes([dp.T, self.Q_pos, dp]) + ca.mtimes([dq.T, self.Q_yaw, dq]) + ca.mtimes([(u-u_ref).T, self.R, (u-u_ref)])

    def terminal_cost(self, x, x_ref=None):
        if x_ref is None:
            x_ref = ca.zeros(self.nx)
        # dx = x - x_ref
        # return ca.mtimes([dx.T, self.Qn, dx])
        
        _p_r = x[0:2]
        _p_r_sp = x_ref[0:2]
        _q_r = x[2]
        _q_r_sp = x_ref[2]
        
        dp = _p_r - _p_r_sp
        # https://www.reddit.com/r/gamedev/comments/1onzya/what_is_the_best_way_to_calculate_angle/#:~:text=given:%20x%20=%20(%20a%20%2D,is%20right%20of%20another%20angle.
        dq = ca.atan2(ca.sin(_q_r - _q_r_sp),ca.cos(_q_r - _q_r_sp))
        
        return ca.mtimes([dp.T, self.Qn_pos, dp]) + ca.mtimes([dq.T, self.Qn_yaw, dq])

    def constraints(self):
        return {
            'lbx': ca.DM([-ca.inf, -ca.inf, -ca.inf]),
            'ubx': ca.DM([ ca.inf,  ca.inf,  ca.inf]),
            'lbu': ca.DM([-1.0, -np.deg2rad(45)]),
            'ubu': ca.DM([ 1.0,  np.deg2rad(45)])
        }