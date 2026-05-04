"""
Unconstrained Model Predictive Control Implementation in Python 
- This version is without an observer, that is, it assumes that the
- the state vector is perfectly known
"""

import numpy as np
import cvxpy as cp
from sarah_m import sarah_m

class ModelPredictiveControl(object):
    """
    A,B,C - system matrices
    f -  prediction horizon
    v  - control horizon
    W3 - input weight matrix
    W4 - prediction weight matrix
    x0 - initial state of the system
    desiredControlTrajectoryTotal - total desired control trajectory
    K - LQR feedback gain
    """
    
    def __init__(self,A,B,C,f,v,W3,W4,x0,desiredControlTrajectoryTotal,K,noise_std=0.04,
                 n_scenarios=20,sarah_b=5,sarah_inner=10,sarah_eta=None,sarah_beta=0.1,sarah_epochs=5,
                 use_sarah_m=True):
        # initialize variables
        self.A=A 
        self.B=B
        self.C=C
        self.f=f
        self.v=v
        self.W3=W3 
        self.W4=W4
        self.desiredControlTrajectoryTotal=desiredControlTrajectoryTotal
        self.K=K
        self.noise_std=noise_std
        self.n_scenarios=n_scenarios
        self.sarah_b=sarah_b
        self.sarah_inner=sarah_inner
        self.sarah_beta=sarah_beta
        self.sarah_epochs=sarah_epochs
        self.use_sarah_m=use_sarah_m

        # dimensions of the matrices
        self.n=A.shape[0]
        self.r=C.shape[0]
        self.m=B.shape[1]        
        
        # this variable is used to track the current time step k of the controller
        # after every calculation of the control inpu, this variables is incremented for +1 
        self.currentTimeStep=0
        
        # the state vectors of the controlled state trajectory
        self.states=[]
        self.states.append(x0)

        # nominal state for center of tube-based robust MPC
        self.nominal_states = [x0]
        
        # the computed inputs 
        self.inputs=[]
        
        # the output vectors of the controlled state trajectory
        self.outputs=[]
        
        
        # form the lifted system matrices and vectors
        # the gain matrix is used to compute the solution 
        # here we pre-compute it to save computational time
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()

        # auto-compute step size from Lipschitz constant (2*lambda_max(H)) if not specified
        if sarah_eta is None:
            lam_max = float(np.max(np.linalg.eigvalsh(np.asarray(self.H_mat))))
            self.sarah_eta = 0.5 / lam_max
        else:
            self.sarah_eta = sarah_eta

        # robust controller uses 99.9% confidence bound
        self.d_max_robust = 3*noise_std

        self.margin_stoch_y = self.calculate_output_stoch_margin(percentile=95)
        self.margin_robust_y = self.calculate_output_robust_margin(d_max=self.d_max_robust)
        print(f"Stochastic Constraint Tightening Margin: {self.margin_stoch_y:.3f}")
        print(f"Robust Constraint Tightening Margin: {self.margin_robust_y:.3f}")

    # this function forms the lifted matrices O and M, as well as the 
    # the gain matrix of the control algorithm and returns them 
    def formLiftedMatrices(self):
        f=self.f
        v=self.v
        r=self.r
        n=self.n
        m=self.m
        A=self.A
        B=self.B
        C=self.C
        
        # lifted matrix O
        O=np.zeros(shape=(f*r,n))

        for i in range(f):
            if (i == 0):
                powA=A;
            else:
                powA=np.matmul(powA,A)
            O[i*r:(i+1)*r,:]=np.matmul(C,powA)

        # lifted matrix M        
        M=np.zeros(shape=(f*r,v*m))
    
        for i in range(f):
            # until the control horizon
            if (i<v):
                for j in range(i+1):
                    if (j == 0):
                        powA=np.eye(n,n);
                    else:
                        powA=np.matmul(powA,A)
                    M[i*r:(i+1)*r,(i-j)*m:(i-j+1)*m]=np.matmul(C,np.matmul(powA,B))
            
            # from control horizon until the prediction horizon
            else:
                for j in range(v):
                    # here we form the last entry
                    if j==0:
                        sumLast=np.zeros(shape=(n,n))
                        for s in range(i-v+2):
                            if (s == 0):
                                powA=np.eye(n,n);
                            else:
                                powA=np.matmul(powA,A)
                            sumLast=sumLast+powA
                        M[i*r:(i+1)*r,(v-1)*m:(v)*m]=np.matmul(C,np.matmul(sumLast,B))
                    else:
                        powA=np.matmul(powA,A)
                        M[i*r:(i+1)*r,(v-1-j)*m:(v-j)*m]=np.matmul(C,np.matmul(powA,B))
        
        tmp1=np.matmul(M.T,np.matmul(self.W4,M))
        self.H_mat   = tmp1 + self.W3                  # (v*m, v*m) — Hessian of MPC cost
        self.MTW4_mat = np.matmul(M.T, self.W4)        # (v*m, f*r)
        tmp2=np.linalg.inv(self.H_mat)
        gainMatrix=np.matmul(tmp2, self.MTW4_mat)

        return O,M,gainMatrix
    

    # this function propagates the dynamics
    # x_{k+1}=Ax_{k}+Bu_{k}
    def propagateDynamics(self,controlInput,state):
        xkp1=np.zeros(shape=(self.n,1))
        yk=np.zeros(shape=(self.r,1))

        # disturbance enters only through the input channel (like a load force)
        w_k = np.matmul(np.asarray(self.B), np.random.normal(0, self.noise_std, size=(self.m, 1)))

        xkp1=np.matmul(self.A,state)+np.matmul(self.B,controlInput) + w_k
        yk=np.matmul(self.C,state)
        
        return xkp1,yk
        

    # this function computes the control inputs, applies them to the system 
    # by calling the propagateDynamics() function and appends the lists
    # that store the inputs, outputs, states
    # def computeControlInputs(self):
                
    #     # extract the segment of the desired control trajectory
    #     desiredControlTrajectory=self.desiredControlTrajectoryTotal[self.currentTimeStep:self.currentTimeStep+self.f,:]

    #     # compute the vector s
    #     vectorS=desiredControlTrajectory-np.matmul(self.O,self.states[self.currentTimeStep])
       
    #     # compute the control sequence
    #     inputSequenceComputed=np.matmul(self.gainMatrix,vectorS)
    #     inputApplied=np.zeros(shape=(1,1))
    #     inputApplied[0,0]=inputSequenceComputed[0,0]
        
    #     # compute the next state and output
    #     state_kp1,output_k=self.propagateDynamics(inputApplied,self.states[self.currentTimeStep])
        
    #     # append the lists
    #     self.states.append(state_kp1)
    #     self.outputs.append(output_k)
    #     self.inputs.append(inputApplied)
    #     # increment the time step
    #     self.currentTimeStep=self.currentTimeStep+1

    def computeControlInputs(self):
        # extract the segment of the desired control trajectory
        Y_LIMIT = self.desiredControlTrajectoryTotal[self.currentTimeStep:self.currentTimeStep+self.f, :]
        desiredControlTrajectory = np.ones((self.f,1)) * -5.0

        

        # get the current states
        nominal_state = self.nominal_states[-1]
        actual_state = self.states[self.currentTimeStep]

        # solve for the nominal control sequence
        if self.use_sarah_m:
            vectorS = desiredControlTrajectory - np.matmul(self.O, nominal_state)
            dw_init = np.zeros((self.f * self.m, 1))
            
            full_grad, stoch_grad = self.make_scenario_cost(
                nominal_state, desiredControlTrajectory, self.n_scenarios, self.noise_std)
            
            # 1. Run the super fast UNCONSTRAINED SARAH-M
            unconstrained_dw = sarah_m(dw_init, full_grad, stoch_grad,
                                       n=self.n_scenarios, b=self.sarah_b, m=self.sarah_inner,
                                       eta=self.sarah_eta, beta=self.sarah_beta, max_epochs=self.sarah_epochs)
            
            # 2. Project the final result ONCE to respect the Stochastic Tightening
            constrained_dw = self.project_state_constraints(
                unconstrained_dw, self.M, self.O, nominal_state, Y_LIMIT, self.margin_stoch_y)
                
            inputNominal = constrained_dw[0:self.m, :]
        else:
            # 1. Run the super fast UNCONSTRAINED Robust (Analytical)
            vectorS = desiredControlTrajectory - np.matmul(self.O, nominal_state)
            unconstrained_u = np.matmul(self.gainMatrix, vectorS)
            
            # 2. Project the final result ONCE to respect the Robust Tightening
            constrained_u = self.project_state_constraints(
                unconstrained_u, self.M, self.O, nominal_state, Y_LIMIT, self.margin_robust_y)
                
            inputNominal = constrained_u[0:self.m, :]

        inputFeedback=np.matmul(self.K, (actual_state-nominal_state))
        inputApplied=inputNominal + np.asarray(inputFeedback)

        # compute the next state and output
        state_kp1,output_k=self.propagateDynamics(inputApplied,actual_state)
        next_nominal_state = np.matmul(self.A, nominal_state) + np.matmul(self.B, inputNominal)

        # append the lists
        self.states.append(state_kp1)
        self.nominal_states.append(next_nominal_state)
        self.outputs.append(output_k)
        self.inputs.append(inputApplied)

        # increment the time step
        self.currentTimeStep=self.currentTimeStep+1
        
    
    def make_scenario_cost(self, nominal_state, desiredTraj, M_sc, noise_std):
        """
        Build per-scenario gradient closures for the stochastic MPC cost.

        For each scenario j we draw a disturbance sequence w_{j,0}..w_{j,f-1}
        (each entering through B), propagate it through the prediction horizon
        to get D_j (the disturbance contribution to the lifted output vector),
        and define s_j = desiredTraj - O*nominal_state - D_j.

        The per-scenario cost is  f_j(U) = ||M U - s_j||^2_{W4} + ||U||^2_{W3}.
        Its gradient is  2(H U - MTW4 s_j)  where H = M^T W4 M + W3.
        Full / stochastic gradients average over all / a mini-batch of scenarios.
        Scenario generation is vectorized over the M_sc dimension.
        """
        f, r, n_st, m_in = self.f, self.r, self.n, self.m

        A   = np.asarray(self.A)
        B   = np.asarray(self.B)
        C   = np.asarray(self.C)
        O   = np.asarray(self.O)
        K_arr = np.asarray(self.K)
        nom = np.asarray(nominal_state).reshape(n_st, 1)

        # Closed loop state matrix
        A_cl = A + B @ K_arr

        s_nominal = np.asarray(desiredTraj).reshape(f * r, 1) - O @ nom  # (f*r, 1)

        # Draw all scenario noise at once: (f, M_sc, m_in)
        noise_all = np.random.normal(0, noise_std, (f, M_sc, m_in))

        # Propagate disturbances through prediction horizon, vectorized over scenarios.
        # x_dist[j] accumulates the disturbance-induced state for scenario j.
        # Loop is over t (length f=20) only, not over j (M_sc scenarios).
        x_dist = np.zeros((M_sc, n_st))       # (M_sc, n_st)
        D      = np.empty((M_sc, f * r))       # (M_sc, f*r)
        for t in range(f):
            # Propagate using closed-loop error dynamics (A_cl.T) instead of open-loop (A.T)
            x_dist = x_dist @ A_cl.T + noise_all[t] @ B.T   # (M_sc, n_st)
            D[:, t * r:(t + 1) * r] = x_dist @ C.T       # (M_sc, r)

        # s_scenarios[j] = s_nominal - D[j],  shape (M_sc, f*r)
        s_scenarios = s_nominal.squeeze() - D             # broadcast (f*r,) - (M_sc, f*r)

        H    = np.asarray(self.H_mat)
        MTW4 = np.asarray(self.MTW4_mat)

        def full_grad(dw):
            s_avg = s_scenarios.mean(axis=0).reshape(-1, 1)   # (f*r, 1)
            return 2.0 * (H @ dw - MTW4 @ s_avg)

        def stoch_grad(dw, S):
            s_avg = s_scenarios[S].mean(axis=0).reshape(-1, 1)  # (f*r, 1)
            return 2.0 * (H @ dw - MTW4 @ s_avg)

        return full_grad, stoch_grad
    

    def project_box_constraints(w, u_min, u_max):
        return np.clip(w, u_min, u_max)
    

    def calculate_stochastic_margin(self, num_scenarios=2000, horizon=150, percentile=95):
        """
        Simulates the closed-loop error over many scenarios to find the probabilistic 
        95th-percentile bound of the feedback control effort.
        """
        A_cl = self.A + np.matmul(self.B, self.K)
        n_st = self.A.shape[0]
        m_in = self.B.shape[1]
        
        # Initialize error state for all scenarios to zero
        e = np.zeros((num_scenarios, n_st))
        u_fb_history = []
        
        for _ in range(horizon):
            # Generate random noise based on your true standard deviation
            v = np.random.normal(0, self.noise_std, (num_scenarios, m_in))
            
            # Calculate the feedback effort used to fight this error
            u_fb = np.matmul(e, self.K.T) # Shape: (num_scenarios, m_in)
            u_fb_history.append(np.abs(u_fb))
            
            # Propagate the closed-loop error dynamics: e = A_cl*e + B*v
            e = np.matmul(e, A_cl.T) + np.matmul(v, self.B.T)
            
        # Extract the 95th percentile of the maximum control effort applied
        margin = np.percentile(u_fb_history, percentile)
        return margin
    

    def calculate_robust_margin(self, d_max, horizon=150):
        """
        Calculates the absolute worst-case feedback control effort by summing the 
        impulse response of the closed-loop system, assuming the noise acts in 
        the worst possible direction at every single time step.
        """
        A_cl = self.A + np.matmul(self.B, self.K)
        n_st = self.A.shape[0]
        m_in = self.B.shape[1]
        
        margin_robust = 0.0
        A_cl_pow = np.eye(n_st)
        
        for i in range(horizon):
            # Calculate how a disturbance at step i affects the current control effort
            # The matrix mapping disturbance v to feedback input u_fb is: K * A_cl^i * B
            term = np.matmul(self.K, np.matmul(A_cl_pow, self.B))
            
            # Add the absolute worst-case contribution 
            margin_robust += np.sum(np.abs(term)) * d_max
            
            # Advance the closed-loop state transition matrix
            A_cl_pow = np.matmul(A_cl, A_cl_pow)
            
        return margin_robust
    

    def calculate_output_stoch_margin(self, num_scenarios=2000, horizon=150, percentile=95):
        A_cl = self.A + np.matmul(self.B, self.K)
        n_st = self.A.shape[0]
        m_in = self.B.shape[1]
        
        e = np.zeros((num_scenarios, n_st))
        y_err_history = []
        
        for _ in range(horizon):
            v = np.random.normal(0, self.noise_std, (num_scenarios, m_in))
            # Track the output error: y_err = C * e
            y_err = np.matmul(e, self.C.T) 
            y_err_history.append(np.abs(y_err))
            
            e = np.matmul(e, A_cl.T) + np.matmul(v, self.B.T)
            
        return np.percentile(y_err_history, percentile)
    

    def calculate_output_robust_margin(self, d_max, horizon=150):
        A_cl = self.A + np.matmul(self.B, self.K)
        n_st = self.A.shape[0]
        
        margin_robust = 0.0
        A_cl_pow = np.eye(n_st)
        
        for i in range(horizon):
            # The matrix mapping disturbance v to output error y_err is: C * A_cl^i * B
            term = np.matmul(self.C, np.matmul(A_cl_pow, self.B))
            margin_robust += np.sum(np.abs(term)) * d_max
            A_cl_pow = np.matmul(A_cl, A_cl_pow)
            
        return margin_robust
    

    def project_state_constraints(self, w_temp, M_mat, O_mat, nominal_state, Y_limit, margin):
        """
        Projects the control sequence so that the predicted outputs NEVER fall below Y_limit + margin
        """
        n_vars = w_temp.shape[0]
        u = cp.Variable((n_vars, 1))
        
        # We enforce: M*u + O*x0 >= Y_limit + margin
        # Therefore: M*u >= Y_limit + margin - O*x0
        bound = Y_limit + margin - np.matmul(O_mat, nominal_state)
        
        # Minimize the distance between the safe u and the SARAH-M gradient step (w_temp)
        objective = cp.Minimize(cp.sum_squares(u - w_temp))
        constraints = [M_mat @ u >= bound]
        
        prob = cp.Problem(objective, constraints)
        try:
            # OSQP is exceptionally fast for repeated micro-QPs
            prob.solve(solver=cp.OSQP, warm_start=True) 
            if u.value is not None:
                return u.value
        except:
            pass
        
        return w_temp # Fallback in case of numerical solver failure