"""
This is the drive code for the model predictive controller -

Unconstrained Model Predictive Control Implementation in Python 
- This version is without an observer, that is, it assumes that the
- the state vector is perfectly known
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

from functionMPC import systemSimulate
from ModelPredictiveControl import ModelPredictiveControl

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
timeSteps=900

# pulse trajectory
desiredTrajectory=np.zeros(shape=(timeSteps,1))
desiredTrajectory[0:300,:]=np.ones((300,1))
desiredTrajectory[600:,:]=np.ones((300,1))


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
# Closed-loop error dynamics:  e_{k+1} = A_cl e_k + w_k,  w_k = B d_k
# Worst-case output deviation after k steps (l1 gain, adversarial input sign):
#   h_k = d_max * sum_{j=0}^{k-1} |C A_cl^j B|
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
# SARAH-M Monte Carlo — probabilistic tube
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
# Plot — worst-case robust tube vs SARAH-M probabilistic tube
###############################################################################
plt.figure(figsize=(12, 6))

# worst-case band centred on the nominal trajectory
plt.fill_between(t, cf_nominal - robust_hw, cf_nominal + robust_hw,
                 alpha=0.18, color='tab:red',
                 label=f'Robust worst-case tube (99th % per sample)')

# SARAH-M Monte Carlo band
plt.fill_between(t, mc_lower, mc_upper,
                 alpha=0.40, color='steelblue',
                 label=f'SARAH-M probabilistic tube (5th-95th %, N={N_mc})')

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






















