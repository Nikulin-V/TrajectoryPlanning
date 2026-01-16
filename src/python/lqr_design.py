import numpy as np
from scipy.linalg import solve_continuous_are

m = 1.0
J = 0.01
g = 9.81
T_0 = m * g

def build_matrices():
    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1/m, 0],
        [0, 0],
        [0, 1/J]
    ])
    return A, B

def design_lqr(A, B, Q=None, R=None):
    if Q is None:
        Q = np.diag([100, 10, 100, 10, 50, 5])
    if R is None:
        R = np.diag([1, 0.01])
    
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    
    return K, P

def analyze_closed_loop(A, B, K):
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    
    return A_cl, eigvals

if __name__ == '__main__':
    A, B = build_matrices()
    
    Q = np.diag([100, 10, 100, 10, 50, 5])
    R = np.diag([1, 1])
    
    K, P = design_lqr(A, B, Q, R)
    A_cl, eigvals = analyze_closed_loop(A, B, K)
    
    print("Матрица коэффициентов обратной связи K:")
    print(K)
    print("\nСобственные значения замкнутой системы:")
    for i, eig in enumerate(eigvals):
        print(f"λ_{i+1} = {eig:.3f}")
    
    print("\nМатрица замкнутой системы A_cl = A - B*K:")
    print(A_cl)

