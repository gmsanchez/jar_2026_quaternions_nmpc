import numpy as np

import casadi as ca

from nmpc_path_follower.utils.quaternions import quaternion_vector_rotation, quaternion_vector_product
from nmpc_path_follower.MPC_minimal import MPCMinimal

class RigidBodyQuaternions(MPCMinimal):
    def __init__(self, N=15, dt=0.1):
        super().__init__(N, dt, solver_type="ipopt")
    
    @property
    def nx(self):
        return 7
    
    @property
    def nu(self):
        return 6
    
    @property
    def Q_pos(self):
        return ca.diag([5.0, 5.0, 1.0])
    
    @property
    def Q_q(self):
        return ca.diag([10.0, 1.0, 1.0, 10.0])
    
    @property
    def Qn_pos(self):
        return ca.diag([5.0, 5.0, 1.0])
    
    @property
    def Qn_q(self):
        return ca.diag([10.0, 1.0, 1.0, 10.0])
    
    @property
    def R(self):
        return ca.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.1])
    
    def dynamics(self, x, u):
        """Continuous-time ODE model."""
        pos = x[0:3]
        q = x[3:7] * [1, 0, 0, 1]
        lin_vel = u[0:3] * [1, 0, 0]
        ang_vel = u[3:6] * [0, 0, 1]
        dxdt = [quaternion_vector_rotation(q, lin_vel),
                0.5 * quaternion_vector_product(q, ang_vel)]
        return ca.vcat(dxdt)

    def stage_cost(self, x, u, x_ref=None, u_ref=None):
        if x_ref is None:
            x_ref = ca.zeros(self.nx)
        if u_ref is None:
            u_ref = ca.zeros(self.nu)
            
        # dx = x - x_ref
        _du = u - u_ref
        
        _p_r = x[0:3]
        _p_r_sp = x_ref[0:3]
        _q_r = x[3:7]
        _q_r_sp = x_ref[3:7]
        
        q_e_r = 1.0 - ca.fabs (ca.dot(_q_r, _q_r_sp))
        q_norm = (1.0 - ca.norm_2(_q_r)**2)**2
        
        _l_cost = 0
        _l_cost +=  q_norm
        _l_cost += ca.mtimes([_du.T, self.R, _du])
        _l_cost += ca.mtimes([(_p_r - _p_r_sp).T, self.Q_pos, (_p_r - _p_r_sp)])
        _l_cost += ca.mtimes([q_e_r.T, q_e_r])
        return _l_cost

    def terminal_cost(self, x, x_ref=None):
        if x_ref is None:
            x_ref = ca.zeros(self.nx)
            
        _p_r = x[0:3]
        _p_r_sp = x_ref[0:3]
        _q_r = x[3:7]
        _q_r_sp = x_ref[3:7]
        
        q_e_r = 1.0 - ca.fabs (ca.dot(_q_r, _q_r_sp))
        q_norm = (1.0 - ca.norm_2(_q_r)**2)**2
        
        _t_cost = 0
        _t_cost +=  q_norm
        _t_cost += ca.mtimes([(_p_r - _p_r_sp).T, self.Qn_pos, (_p_r - _p_r_sp)])
        _t_cost += ca.mtimes([q_e_r.T, q_e_r])
        return _t_cost


    def constraints(self):
        return {
            'lbx': ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf, -ca.inf, -ca.inf, -ca.inf]),
            'ubx': ca.DM([ ca.inf,  ca.inf,  ca.inf,  ca.inf,  ca.inf,  ca.inf,  ca.inf]),
            'lbu': ca.DM([-1.0, 0.0, 0.0, np.deg2rad(0), np.deg2rad(0), -np.deg2rad(180)]),
            'ubu': ca.DM([ 1.0, 0.0, 0.0, np.deg2rad(0), np.deg2rad(0),  np.deg2rad(180)])
        }
