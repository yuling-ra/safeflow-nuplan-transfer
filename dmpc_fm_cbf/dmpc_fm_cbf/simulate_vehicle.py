import numpy as np

def simulate_vehicle(x0, u_seq, dt=0.1):
    """
    x0: [x, y, theta, v, r]
    u_seq: (K, 2) where each is [a, delta]
    returns: (K+1, 5) states
    """
    u_seq = np.asarray(u_seq, dtype=float)
    K = u_seq.shape[0]

    states = np.zeros((K + 1, 5), dtype=float)
    states[0] = np.asarray(x0, dtype=float)

    for k in range(K):
        x, y, theta, v, r = states[k]
        a, delta = u_seq[k]

        x_next     = x + dt * v * np.cos(theta)
        y_next     = y + dt * v * np.sin(theta)
        theta_next = theta + dt * r
        v_next     = v + dt * a
        r_next     = r + dt * delta

        states[k+1] = [x_next, y_next, theta_next, v_next, r_next]

    return states
