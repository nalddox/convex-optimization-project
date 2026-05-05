"""
This is the drive code for the model predictive controller -

Unconstrained Model Predictive Control Implementation in Python 
- This version is without an observer, that is, it assumes that the
- the state vector is perfectly known
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

from functionMPC import systemSimulate
from ModelPredictiveControl import ModelPredictiveControl
from sarah_m import sarah_m

###############################################################################
#  Define the MPC algorithm parameters
###############################################################################
# prediction horizon
f= 20

# control horizon 
v=20


###############################################################################
# Define the model
###############################################################################
# masses, spring and damper constants
m1=2  ; m2=2   ; k1=100  ; k2=200 ; d1=1  ; d2=5; 

# define the continuous-time system matrices
Ac=np.matrix([[0, 1, 0, 0],
              [-(k1+k2)/m1 ,  -(d1+d2)/m1 , k2/m1 , d2/m1 ],
              [0 , 0 ,  0 , 1], 
              [k2/m2,  d2/m2, -k2/m2, -d2/m2]])
Bc=np.matrix([[0],[0],[0],[1/m2]])
Cc=np.matrix([[1, 0, 0, 0]])

r=1; m=1 # number of inputs and outputs
n= 4 # state dimension


###############################################################################
# discretize and simulate the system step response
###############################################################################
# discretization constant
sampling=0.05

# model discretization
I=np.identity(Ac.shape[0]) # this is an identity matrix
A=np.linalg.inv(I-sampling*Ac)
B=A*sampling*Bc
C=Cc

# check the eigenvalues
eigen_A=np.linalg.eig(Ac)[0]
eigen_Aid=np.linalg.eig(A)[0]

timeSampleTest=200

# compute the system's step response
inputTest=10*np.ones((1,timeSampleTest))
x0test=np.zeros(shape=(4,1))

# simulate the discrete-time system 
Ytest, Xtest=systemSimulate(A,B,C,inputTest,x0test)

# plt.figure(figsize=(8,8))
# plt.plot(Ytest[0,:],linewidth=4, label='Step response - output')
# plt.xlabel('time steps')
# plt.ylabel('output')
# plt.legend()
# plt.savefig('stepResponse.png',dpi=600)
# plt.show()


###############################################################################
# form the weighting matrices
###############################################################################
# W1 matrix
W1=np.zeros(shape=(v*m,v*m))

for i in range(v):
    if (i==0):
        W1[i*m:(i+1)*m,i*m:(i+1)*m]=np.eye(m,m)
    else:
        W1[i*m:(i+1)*m,i*m:(i+1)*m]=np.eye(m,m)
        W1[i*m:(i+1)*m,(i-1)*m:(i)*m]=-np.eye(m,m)

# W2 matrix
Q0=0.0000000011
Qother=0.0001

W2=np.zeros(shape=(v*m,v*m))

for i in range(v):
    if (i==0):
        W2[i*m:(i+1)*m,i*m:(i+1)*m]=Q0
    else:
        W2[i*m:(i+1)*m,i*m:(i+1)*m]=Qother

# W3 matrix        
W3=np.matmul(W1.T,np.matmul(W2,W1))

# W4 matrix
W4=np.zeros(shape=(f*r,f*r))

# in the general case, this constant should be a matrix
predWeight=10

for i in range(f):
    W4[i*r:(i+1)*r,i*r:(i+1)*r]=predWeight


###############################################################################
# Define the reference trajectory 
###############################################################################
timeSteps=300

# pulse trajectory
desiredTrajectory=np.zeros(shape=(timeSteps,1))
desiredTrajectory[0:100,:]=np.ones((100,1))
desiredTrajectory[200:,:]=np.ones((100,1))


###############################################################################
# Define the LQR feedback gain 
###############################################################################
# Solve the Discrete Algebraic Riccati Equation (DARE)
Q=np.eye(n) * 1    # weight on state error
R=np.eye(r) * 10     # weight on ancillary control effort
P=solve_discrete_are(A,B,Q,R)

# Compute LQR gain K
K=-np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)


###############################################################################
# Simulate the MPC algorithm
###############################################################################
T = timeSteps - f
t = np.arange(T)
x0 = x0test
noise_std = 5.0   # force-channel noise

# Closed-form run: fast, gives the noise-free nominal trajectory used to centre
# the worst-case tube and as the legend reference.
mpc_cf = ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory,K,
                                noise_std=noise_std, use_sarah_m=False)
for i in range(T):
    mpc_cf.computeControlInputs()

desiredList = [desiredTrajectory[j,0]              for j in range(T)]
cf_nominal  = np.array([float(C @ mpc_cf.nominal_states[j]) for j in range(T)])


###############################################################################
# Worst-case robust tube (analytical)
###############################################################################
A_arr = np.asarray(A);  B_arr = np.asarray(B)
C_arr = np.asarray(C);  K_arr = np.asarray(K)
A_cl  = A_arr + B_arr @ K_arr

d_max      = 3 * noise_std  # 90th-percentile bound per sample, matches SARAH-M 5th-95th level
A_cl_pow   = np.eye(n)
acc        = 0.0
robust_hw  = np.zeros(T)       # half-width of worst-case tube at each step
for k in range(T):
    robust_hw[k] = acc
    acc       += abs(float(C_arr @ A_cl_pow @ B_arr)) * d_max
    A_cl_pow   = A_cl_pow @ A_cl


###############################################################################
# SARAH-M Monte Carlo tube
###############################################################################
N_mc = 20
mc_outputs = np.zeros((N_mc, T))
for trial in range(N_mc):
    mpc_trial = ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory,K,
                                       noise_std=noise_std, use_sarah_m=True)
    for i in range(T):
        mpc_trial.computeControlInputs()
    mc_outputs[trial, :] = [mpc_trial.outputs[j][0,0] for j in range(T)]

mc_mean  = np.mean(mc_outputs,   axis=0)
mc_lower = np.percentile(mc_outputs,  5, axis=0)
mc_upper = np.percentile(mc_outputs, 95, axis=0)


###############################################################################
# Plot — worst-case robust tube vs stochastic tube
###############################################################################
plt.figure(figsize=(12, 6))

# worst-case band centred on the nominal trajectory
plt.fill_between(t, cf_nominal - robust_hw, cf_nominal + robust_hw,
                 alpha=0.18, color='tab:red',
                 label=f'Robust worst-case tube (99th%)')

# SARAH-M Monte Carlo band
plt.fill_between(t, mc_lower, mc_upper,
                 alpha=0.40, color='steelblue',
                 label=f'SARAH-M tube (5th-95th%)')

plt.plot(mc_mean,    linewidth=1.5, color='steelblue', label='SARAH-M mean trajectory')
plt.plot(cf_nominal, '--', color='orange', linewidth=2.0,
         label='Nominal trajectory (robust tube center)', zorder=5)
plt.plot(desiredList,'r',  linewidth=2.0, label='Desired', zorder=6)

plt.xlabel('time steps')
plt.ylabel('Output')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('tube_comparison.png', dpi=600)
plt.show()


###############################################################################
# SARAH-M gradient convergence (single representative time step, t=0)
###############################################################################
# mpc_diag = ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory,K,
#                                    noise_std=noise_std, use_sarah_m=True)

# full_grad_diag, stoch_grad_diag = mpc_diag.make_scenario_cost(
#     x0, np.ones((f, 1)) * -5.0, mpc_diag.n_scenarios, noise_std)

# dw_init_diag = np.zeros((f * mpc_diag.m, 1))
# _, grad_history = sarah_m(dw_init_diag, full_grad_diag, stoch_grad_diag,
#                            n=mpc_diag.n_scenarios, b=mpc_diag.sarah_b,
#                            m=mpc_diag.sarah_inner, eta=mpc_diag.sarah_eta,
#                            beta=mpc_diag.sarah_beta, max_epochs=mpc_diag.sarah_epochs,
#                            return_grad_history=True)

# # Mark epoch boundaries: each epoch has 1 full-grad eval + sarah_inner inner steps
# epoch_len = mpc_diag.sarah_inner + 1
# epoch_starts = [s * epoch_len for s in range(mpc_diag.sarah_epochs)]

# plt.figure(figsize=(8, 5))
# plt.semilogy(grad_history, linewidth=2, color='steelblue', label='||G_k||')
# for idx, es in enumerate(epoch_starts):
#     plt.axvline(es, color='gray', linestyle='--', linewidth=0.8,
#                 label='Epoch start' if idx == 0 else None)
# plt.xlabel('SARAH-M iteration')
# plt.ylabel('Gradient norm $\\|G_k\\|$')
# plt.title('SARAH-M Gradient Convergence (t = 0)')
# plt.legend()
# plt.grid(True, which='both', alpha=0.4)
# plt.tight_layout()
# plt.savefig('stochastic_convergence.png', dpi=600)
# plt.show()


###############################################################################
# Runtime comparison: Robust (closed-form) vs SARAH-M
###############################################################################
T_time = 150  # steps to time

times_robust = []
mpc_rt_robust = ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory,K,
                                        noise_std=noise_std, use_sarah_m=False)
for i in range(T_time):
    t0 = time.perf_counter()
    mpc_rt_robust.computeControlInputs()
    times_robust.append(time.perf_counter() - t0)

times_sarah = []
mpc_rt_sarah = ModelPredictiveControl(A,B,C,f,v,W3,W4,x0,desiredTrajectory,K,
                                       noise_std=noise_std, use_sarah_m=True)
for i in range(T_time):
    t0 = time.perf_counter()
    mpc_rt_sarah.computeControlInputs()
    times_sarah.append(time.perf_counter() - t0)

times_robust = np.array(times_robust) * 1e3  # convert to ms
times_sarah  = np.array(times_sarah)  * 1e3

steps = np.arange(T_time)
plt.figure(figsize=(9, 5))
plt.plot(steps, times_robust, color='tab:orange', linewidth=1.2,
         label=f'Robust  (mean = {times_robust.mean():.2f} ms)')
plt.plot(steps, times_sarah,  color='steelblue', linewidth=1.2,
         label=f'SARAH-M (mean = {times_sarah.mean():.2f} ms)')
plt.axhline(times_robust.mean(), color='tab:orange', linestyle='--', linewidth=0.8)
plt.axhline(times_sarah.mean(),  color='steelblue',  linestyle='--', linewidth=0.8)
plt.xlabel('step')
plt.ylabel('time (ms)')
plt.title(f'Per-step computation time over {T_time} MPC steps')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('runtime_comparison.png', dpi=600)
plt.show()






















