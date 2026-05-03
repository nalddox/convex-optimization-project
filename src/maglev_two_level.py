import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

# --- 1. Define Constants & Stabilized System ---
m = 750.0
Ts = 0.01  # CHANGED: 10ms sample time gives a 300ms prediction horizon

A1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
B1 = np.array([[0], [0], [1]])
C1 = np.array([[1, 0, 0]])
K = np.array([[-172980, -7660, -220]])

# Stabilized continuous system
A_stab = A1 + B1 @ K
B_stab = B1
C_stab = C1
D_stab = np.array([[0]])

# Disturbance matrix (Force enters acceleration)
G = np.array([[0], [1 / m], [0]])

# Discretize
sys_discrete = cont2discrete((A_stab, B_stab, C_stab, D_stab), Ts, method='zoh')
Ad, Bd, Cd = sys_discrete[0], sys_discrete[1], sys_discrete[2]
Gd = cont2discrete((A_stab, G, C_stab, D_stab), Ts, method='zoh')[1]

# --- 2. MPC Parameters & Matrices ---
P = 30
M = 4
Q = np.eye(P)
R_delta = np.exp(-10) * np.eye(M)

# CHANGED: Reference is now 0 (deviation from 10mm equilibrium)
Y_r = np.zeros((P, 1))

Psi = np.zeros((P, 3))
Theta = np.zeros((P, M))
Gamma_w = np.zeros((P, 1))
Gamma_G = np.zeros((P, 1))

def get_step_response(A, B, C, n):
    val = np.zeros((C.shape[0], B.shape[1]))
    for k in range(n + 1):
        val += C @ np.linalg.matrix_power(A, k) @ B
    return val[0, 0]

for i in range(P):
    Psi[i, :] = Cd @ np.linalg.matrix_power(Ad, i + 1)
    Gamma_w[i, 0] = get_step_response(Ad, Bd, Cd, i)
    Gamma_G[i, 0] = get_step_response(Ad, Gd, Cd, i)
    for j in range(M):
        if i >= j:
            Theta[i, j] = get_step_response(Ad, Bd, Cd, i - j)

H = 2 * (Theta.T @ Q @ Theta + R_delta)
H_inv = np.linalg.inv(H)

# --- 3. Simulation Loop ---
sim_time = 1.0  
N_steps = int(sim_time / Ts)
t_array = np.arange(0, sim_time, Ts)

y_history = np.zeros(N_steps)
w_history = np.zeros(N_steps)
d_history = np.zeros(N_steps)

# CHANGED: Start at [0,0,0] which represents the 10mm equilibrium
x = np.array([[0.0], [0.0], [0.0]])  
w_prev = 0.0
d_prev = 0.0

for k in range(N_steps):
    # Disturbance hits at 0.5s
    d_curr = 1000.0 if t_array[k] >= 0.5 else 0.0
        
    # Log absolute air gap (deviation + 10mm)
    y_history[k] = x[0, 0] + 0.01 
    d_history[k] = d_curr
    
    # MPC Prediction & Optimization
    Es = Y_r - (Psi @ x) - (Gamma_w * w_prev) - (Gamma_G * d_prev)
    f = -2 * Theta.T @ Q @ Es
    Delta_w_opt = -H_inv @ f
    
    # Receding Horizon
    dw = Delta_w_opt[0, 0]
    w_curr = w_prev + dw
    w_history[k] = w_curr
    
    # Plant Update
    x = Ad @ x + (Bd * w_curr) + (Gd * d_curr)
    
    w_prev = w_curr
    d_prev = d_curr

# --- 4. Plotting ---
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_array, y_history * 1000, 'b-', linewidth=2)
plt.axhline(y=10, color='r', linestyle='--', label='Reference (10 mm)')
plt.ylabel('Air Gap (mm)')
plt.title('Maglev System MPC Response')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_array, w_history, 'g-', linewidth=2)
plt.ylabel('Control Input $w(k)$')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_array, d_history, 'k-', linewidth=2)
plt.ylabel('Disturbance Force (N)')
plt.xlabel('Time (seconds)')
plt.grid(True)

plt.tight_layout()
plt.show()