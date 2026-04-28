import casadi as ca
import numpy as np
from abc import ABC, abstractmethod

import time
import matplotlib.pyplot as plt


class IndexMap:
    def __init__(self, N, nx, nu):
        self.idx = {}
        off = 0
        for k in range(N):
            self.idx[('x', k)] = slice(off, off+nx); off += nx
            self.idx[('u', k)] = slice(off, off+nu); off += nu
        self.idx[('x', N)] = slice(off, off+nx)
        self.nw = off + nx

    def x(self, k): return self.idx[('x', k)]
    def u(self, k): return self.idx[('u', k)]


class MPCMinimal(ABC):
    
    def __init__(self, N, dt, solver_type="ipopt"):
        self.N = N
        self.dt = dt
        
        self.solver_type = solver_type
        
        self.opti = ca.Opti()
        self.X, self.U = [], []         # Decision Variables
        self.X0_p = None                # Parameter: Initial State
        self.X_ref_p, self.U_ref_p = [], [] # Parameters: References
                
        # Initialize numerical placeholders for the OCP
        self.current_x0 = np.zeros((self.nx,))
        self.current_x_ref = np.zeros((self.nx, self.N+1))
        self.current_u_ref = np.zeros((self.nu, self.N))
        self.current_z0 = np.zeros(self.nx*(self.N+1) + self.nu*self.N)
        self.current_p0 = ca.vertcat(ca.DM.zeros(self.nx),
                                     ca.vec(ca.DM.zeros(self.nx, self.N+1)),
                                     ca.vec(ca.DM.zeros(self.nu, self.N)))
        
        # Integration helper (will be built in setup)
        self._integrator_func = None  ## TODO
        self.current_elapsed_time = 0.0
        
        # Index map for easy indexing of optimization variables
        self.idx = IndexMap(self.N, self.nx, self.nu)
        
        self._is_setup = False

        
    @property
    @abstractmethod
    def nx(self):
        pass
    
    @property
    @abstractmethod
    def nu(self):
        pass
    
    @abstractmethod
    def dynamics(self, x, u):
        """
        Continuous-time dynamics: x_dot = dynamics(x,u)
        """
        pass
    
    @abstractmethod
    def stage_cost(self, x, u, x_ref=None, u_ref=None):
        pass

    @abstractmethod
    def terminal_cost(self, x, x_ref=None):
        pass

    @abstractmethod
    def constraints(self):
        """
        Returns a generic definition of constraints.
        Format: {'lbx': ..., 'ubx': ..., 'lbu': ..., 'ubu': ...}
        """
        pass

    def __rk4(self, x, u):
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + self.dt/2*k1, u)
        k3 = self.dynamics(x + self.dt/2*k2, u)
        k4 = self.dynamics(x + self.dt*k3, u)
        xrk4 = x + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return ca.vertcat(xrk4)

    def __build_integrator_func(self):
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        
        ode_integrator = dict(x=x, p=u, ode=self.dynamics(x, u))
        intoptions = {
            "abstol": 1e-8,
            "reltol": 1e-8,
        }
        return ca.integrator("int_ode", "cvodes", ode_integrator, 0, self.dt, intoptions)

    def simulate(self, x, u):
        vdpargs = dict(x0=x, p=u)
        out = self.__integrator_func(**vdpargs)
        return np.array(out["xf"]).flatten()
          
    def setup(self):
        """
        Setup the CasADi + IPOPT/Fatrop solver.
        """
        # 1. Discretize dynamics (e.g., RK4) using self.dynamics()
        # 2. Loop N times to build the graph
        # 3. Apply self.constraints() to the Opti variables
        # 4. Instantiate the solver
        
        if self._is_setup:
            raise RuntimeError("MPC is already setup. Create a new instance.")
        
        self.X0_p = self.opti.parameter(self.nx)
        self.X_ref_p += [self.opti.parameter(self.nx) for _ in range(self.N+1)]
        self.U_ref_p += [self.opti.parameter(self.nu) for _ in range(self.N)]
        
        for k in range(self.N):
            self.X.append(self.opti.variable(self.nx))
            self.U.append(self.opti.variable(self.nu))
            # self.X_ref_p.append(self.opti.parameter(self.nx))
            # self.U_ref_p.append(self.opti.parameter(self.nu))
        self.X.append(self.opti.variable(self.nx))
        # self.X_ref_p.append(self.opti.parameter(self.nx))
       
        C = self.constraints()
        
        # Objective function
        J = 0
        for k in range(self.N):
            J += self.stage_cost(self.X[k], self.U[k], self.X_ref_p[k], self.U_ref_p[k])
        J += self.terminal_cost(self.X[self.N], self.X_ref_p[self.N])
        self.opti.minimize(J)
        
        # Constraints
        for k in range(self.N):
            # Dynamics constraint
            self.opti.subject_to(self.X[k+1] == self.__rk4(self.X[k], self.U[k]))           
            
            # Initial state constraint
            if k==0:
                self.opti.subject_to(self.X[0] == self.X0_p)
            
            # State bounds
            self.opti.subject_to(self.opti.bounded(C['lbx'], self.X[k], C['ubx']))
            
            # Control bounds
            self.opti.subject_to(self.opti.bounded(C['lbu'], self.U[k], C['ubu']))
            
        #     self.opti.set_initial(self.U[k], ca.DM.zeros(self.nu))
        #     self.opti.set_initial(self.X[k], ca.DM.zeros(self.nx))
        # self.opti.set_initial(self.X[self.N], ca.DM.zeros(self.nx))
        
        # State bounds on the final state
        self.opti.subject_to(self.opti.bounded(C['lbx'], self.X[self.N], C['ubx']))
        
        # FATROP options
        options = {}
        options["print_time"] = False
        options["expand"] = True
        options["print_out"] = False
        options["fatrop"] = {"mu_init": 0.1, "print_level": 0}
        options["structure_detection"] = "auto"
        options["debug"] = False #True
        options["verbose"] = 0
        options["record_time"] = True
        # Create NLP solver
        self.opti.solver("fatrop", options)
        
        # # IPOPT options
        # options = {}
        # options["expand"] = True
        # options["ipopt.print_level"] =  0
        # options["ipopt.max_cpu_time"] = 60
        # options["ipopt.max_iter"] = 1000
        # options["ipopt.linear_solver"] = "ma57"
        # options["print_time"] = False
        # options["record_time"] = True
        # options["expand"] = False
        # # Create NLP solver
        # self.opti.solver('ipopt', options)
                
        self.__integrator_func = self.__build_integrator_func()

        self._is_setup = True

    def set_x0(self, x_measured):
        self.current_x0 = x_measured
        self.__update_p0()
        
    def __update_p0(self):
        self.current_p0 = ca.vertcat(
            ca.vec(self.current_x0),
            ca.vec(self.current_x_ref),
            ca.vec(self.current_u_ref)
        )
        self.__opti_set_p0()
        
    def __shift_z0(self):
        # self.current_z0[0:-self.nx-self.nu] = self.current_z0[self.nx+self.nu:]
        for k in range(self.N):
            self.current_z0[self.idx.x(k)] = self.current_z0[self.idx.x(k+1)]
        for k in range(self.N-1):
            self.current_z0[self.idx.u(k)] = self.current_z0[self.idx.u(k+1)]   
            
    def __opti_set_p0(self):
        self.opti.set_value(self.X0_p, self.current_x0)
        for k in range(self.N):
            self.opti.set_value(self.X_ref_p[k], self.current_x_ref[:,k])
            self.opti.set_value(self.U_ref_p[k], self.current_u_ref[:,k])
        self.opti.set_value(self.X_ref_p[self.N], self.current_x_ref[:,self.N])
        

    def update_reference(self, x_ref, u_ref=None, shift_ref=False):

        x_ref = np.asarray(x_ref)

        # State reference
        if x_ref.ndim == 1:
            if shift_ref:
                x_ref = np.reshape(x_ref, (self.nx, 1))
                x_ref = np.concatenate((self.current_x_ref[:,1:], x_ref), axis=1)
                # self.current_x_ref[:,0:-1] = self.current_x_ref[:,1:]
                # self.current_x_ref[:,-1] = x_ref
            else:
                x_ref = np.tile(x_ref.reshape(-1,1), (1, self.N+1))
        else:
            assert x_ref.shape == (self.nx, self.N+1)

        self.current_x_ref[:,:] = x_ref

        # Control reference
        if u_ref is None:
            u_ref = np.zeros(self.nu)

        u_ref = np.asarray(u_ref)

        if u_ref.ndim == 1:
            if shift_ref:
                u_ref = np.reshape(u_ref, (self.nu, 1))
                u_ref = np.concatenate((self.current_u_ref[:,1:], u_ref), axis=1)
                # self.current_u_ref[:,0:-1] = self.current_u_ref[:,1:]
                # self.current_u_ref[:,-1] = u_ref
            else:
                u_ref = np.tile(u_ref.reshape(-1,1), (1, self.N))
        else:
            assert u_ref.shape == (self.nu, self.N)

        self.current_u_ref[:,:] = u_ref
        
        self.__update_p0()
    


    def solve(self, x_measured, warm_start=True):
        """
        Sets initial state x0, calls solver, returns u_opt.
        """
        
        if not self._is_setup:
            raise RuntimeError("Call setup() before solve().")
        
        self.set_x0(x_measured)
        self.__shift_z0()
        # t0 = time.time()
        if warm_start:
           self.__warm_start()
        sol = self.opti.solve()
        # t1 = time.time()
        # self.current_elapsed_time = t1 - t0
        
        status = self.opti.stats()["unified_return_status"]
        # time_elapsed = -1.0 # self.opti.stats()['fatrop']['time_total']
        time_elapsed = self.opti.stats()['t_wall_total']
        
        # iter_count = self.opti.stats()['fatrop']['iterations_count']
        iter_count = self.opti.stats()["iter_count"]
        # self.current_elapsed_time = t1 - t0
        
        # Store solution
        # for k in range(self.N):
        #     self.current_z0[self.idx.x(k)] = sol.value(self.X[k])
        #     self.current_z0[self.idx.u(k)] = sol.value(self.U[k])
        # self.current_z0[self.idx.x(self.N)] = sol.value(self.X[self.N])
            
        
        self.current_z0 = self.opti.debug.value(self.opti.x)
        # self._last_X = np.column_stack(
        #     [sol.value(self.X[k]) for k in range(self.N+1)]
        # )
        # self._last_U = np.column_stack(
        #     [sol.value(self.U[k]) for k in range(self.N)]
        # )
        
        # print(self.opti.stats()["iter_count"])
        # for k in range(self.N):
        #     self.opti.set_initial(self.X[:], sol.value(self.X[k]))
        # print(self.opti.stats())
        # print(sol.values)
        # print(self.opti.x)
        return (sol.value(self.U[0]), time_elapsed, iter_count, status)
    

    def set_initial_guess(self, x_guess, u_guess):
        # self._last_X = np.column_stack(
        #     [x_guess for k in range(self.N+1)]
        # )
        # self._last_U = np.column_stack(
        #     [u_guess for k in range(self.N)]
        # )
        for k in range(self.N):
            self.current_z0[self.idx.x(k)] = x_guess
            self.current_z0[self.idx.u(k)] = u_guess
        self.current_z0[self.idx.x(self.N)] = x_guess
        self.__opti_set_z0()
        
        
    def __opti_set_z0(self):
        # for k in range(self.N):
        #     self.opti.set_initial(self.X[k], self.current_z0[self.idx.x(k)])
        #     self.opti.set_initial(self.U[k], self.current_z0[self.idx.u(k)])
        # self.opti.set_initial(self.X[self.N], self.current_z0[self.idx.x(self.N)])
        
        # self.__warm_start()
        self.opti.set_initial(self.opti.x, self.current_z0)
    
    def __shift_z0(self):
        # self.current_z0[0:-self.nx-self.nu] = self.current_z0[self.nx+self.nu:]
        for k in range(self.N):
            self.current_z0[self.idx.x(k)] = self.current_z0[self.idx.x(k+1)]
        for k in range(self.N-1):
            self.current_z0[self.idx.u(k)] = self.current_z0[self.idx.u(k+1)] 
    

    def __warm_start(self):
        # if not hasattr(self, "_last_X"):
        #     return

        # # Shift states
        # X_guess = np.hstack([self._last_X[:,1:], self._last_X[:,-1:]])
        # # Shift controls
        # U_guess = np.hstack([self._last_U[:,1:], self._last_U[:,-1:]])

        #X_guess = np.zeros_like(self._last_X)
        #X_guess[:, :-1] = self._last_X[:, 1:]
        #X_guess[:, -1] = self._last_X[:, -1]

        # Shift controls
        #U_guess = np.zeros_like(self._last_U)
        #U_guess[:, :-1] = self._last_U[:, 1:]
        #U_guess[:, -1] = self._last_U[:, -1]

        # Set initial guesses
        # for k in range(self.N):
        #     self.opti.set_initial(self.X[k], X_guess[:, k])
        #     self.opti.set_initial(self.U[k], U_guess[:, k])

        # self.opti.set_initial(self.X[self.N], X_guess[:, self.N])
        # self.__shift_z0()
        self.__opti_set_z0()
        
    
    
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
        
        return ca.mtimes([dp.T, self.Q_pos, dp]) + ca.mtimes([dq.T, self.Qn_yaw, dq]) + ca.mtimes([(u-u_ref).T, self.R, (u-u_ref)])

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
        
        return ca.mtimes([dp.T, self.Q_pos, dp]) + ca.mtimes([dq.T, self.Qn_yaw, dq])

    def constraints(self):
        return {
            'lbx': ca.DM([-ca.inf, -ca.inf, -ca.inf]),
            'ubx': ca.DM([ ca.inf,  ca.inf,  ca.inf]),
            'lbu': ca.DM([-1.0, -np.deg2rad(45)]),
            'ubu': ca.DM([ 1.0,  np.deg2rad(45)])
        }
        
def main():
    print("This is the main script. We will run a MPC on a 2-D particle.")
    plot_results = True
    
    N = 15
    dt = 0.1
    x0 = np.array([0,0,0])
    u0 = np.array([0,0])
    x_sp = np.array([5.0, -5.0, np.deg2rad(0.0)])
    
    mpc = RigidBody2DEuler(N, dt)
    mpc.setup()
    mpc.set_x0(x0)
    mpc.set_initial_guess(x0, u0)
    # Update Reference
    mpc.update_reference(x_sp)
    
    # _x = ca.SX.sym("x", mpc.nx)
    # _u = ca.SX.sym("u", mpc.nu)
    
    # ode_integrator = dict(x=_x, p=_u, ode=mpc.dynamics(_x, _u))
    # intoptions = {
    #     "abstol": 1e-8,
    #     "reltol": 1e-8,
    # }
    # vdp = ca.integrator("int_ode", "cvodes", ode_integrator, 0, dt, intoptions)
    
    Nsim = 200
    times = dt*Nsim*np.linspace(0, 1, Nsim+1)
    x = np.zeros((Nsim+1, mpc.nx))
    x[0, :] = x0
    u = np.zeros((Nsim, mpc.nu))
    u[0, :] = u0
    iter_time = np.zeros((Nsim, 1))
    iter_count = np.zeros((Nsim, 1))

    for t in range(Nsim):
        t0 = time.time()
        U_sol, t_e, i_c, status = mpc.solve(x[t,:])
        t1 = time.time()
        
        # Print stats
        print ("%d: %s %d iterations in %.6f seconds" % (t,status, i_c, t_e))
        # Store solution
        u[t, :] = U_sol
        # Store elapsed time
        iter_time[t] = t_e
        iter_count[t] = i_c
        
        # Simulate
        # vdpargs = dict(x0=x[t, :], p=u[t, :])
        # out = vdp(**vdpargs)
        # x[t+1, :] = np.array(out["xf"]).flatten()
        x[t+1, :] = mpc.simulate(x[t, :], u[t, :])
    
    print(f'Tiempo promedio de ejecucion: {np.mean(iter_time[:-1])}')
    print(f'Iteraciones promedio del solver: {np.mean(iter_count[:-1])}')
    
    Nx, Nu = mpc.nx, mpc.nu
    if plot_results:
        # Plots.
        fig = plt.figure()
        numrows = max(Nx,Nu)
        numcols = 2

        # u plots. Need to repeat last element
        # for stairstep plot.
        u = np.concatenate((u,u[-1:,:]))
        for i in range(Nu):
            ax = fig.add_subplot(numrows,
                numcols,numcols*(i+1))
            ax.step(times,u[:,i],"-k",where="post")
            ax.set_xlabel("Time")
            ax.set_ylabel("Control %d" % (i + 1))
            ax.grid()

        # x plots.    
        for i in range(Nx):
            ax = fig.add_subplot(numrows,
                numcols,numcols*(i+1) - 1)
            ax.plot(times,x[:,i],"-k",label="System")
            ax.set_xlabel("Time")
            ax.set_ylabel("State %d" % (i + 1))
            ax.grid()

        ax = fig.add_subplot(numrows,numcols,numrows*numcols)
        ax.plot(iter_time[1:]*1E3,".-k")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Execution time [ms]")
        ax.grid()
        fig.tight_layout(pad=.5)
        plt.show()
    
    return

        
if __name__== "__main__":
    main()
    