import numpy as np

# SARAH-M: Stochastic Recursive Gradient Algorithm with Momentum
# Algorithm 2 from Yang, Z. (2024) SARAH-M: A fast stochastic recursive gradient descent algorithm via momentum
def sarah_m(w_init, full_grad_func, stoch_grad_func, n, b, m, eta, beta, max_epochs):
    """
    w_init          : Initial parameter vector
    full_grad_func  : Function to compute the full gradient over all n samples
    stoch_grad_func : Function to compute the gradient over a mini-batch S
    n               : Total number of samples/scenarios
    b               : Mini-batch size
    m               : Update frequency (number of inner loop iterations)
    eta             : Learning rate (step size)
    beta            : Momentum coefficient (can be a scalar or sequence, using scalar here)
    max_epochs      : Number of outer loop iterations (s)
    """

    w = np.copy(w_init)
    
    # Outer loop s = 1, 2, ...
    for s in range(max_epochs):
        v_0 = np.copy(w)
        
        # Compute full gradient at the start of the epoch
        G_0 = full_grad_func(v_0)
        
        # Initial updates for the inner loop
        w_1 = v_0 - eta * G_0
        v_1 = np.copy(w_1) # Because w_0 is technically undefined for the first momentum step, 
                           # the paper initializes v_1 = w_1 = v_0 - eta * G_0
        v_k = np.copy(v_1)
        w_k = np.copy(w_1)
        w_k_minus_1 = np.copy(v_0)
        G_k_minus_1 = np.copy(G_0)
        v_k_minus_1 = np.copy(v_0)
        
        # Inner loop k = 1, ..., m
        for k in range(1, m + 1):
            # Randomly choose a mini-batch S of size b
            S = np.random.choice(n, b, replace=False)
            
            # Compute stochastic gradients for the variance reduction estimator
            grad_S_vk = stoch_grad_func(v_k, S)
            grad_S_vk_minus_1 = stoch_grad_func(v_k_minus_1 if k > 1 else v_0, S)
            
            # Recursive gradient estimator update
            G_k = grad_S_vk - grad_S_vk_minus_1 + G_k_minus_1
            
            # Parameter update
            w_k_plus_1 = v_k - eta * G_k
            
            # Momentum update
            v_k_plus_1 = w_k_plus_1 + beta * (w_k_plus_1 - w_k)
            
            # Shift variables for the next iteration
            w_k_minus_1 = np.copy(w_k)
            w_k = np.copy(w_k_plus_1)
            v_k_minus_1 = np.copy(v_k)
            v_k = np.copy(v_k_plus_1)
            G_k_minus_1 = np.copy(G_k)
            
        # Set the outer loop variable to the last iterate of the inner loop
        w = np.copy(w_k)
        
    return w