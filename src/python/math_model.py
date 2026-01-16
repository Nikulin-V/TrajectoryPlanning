import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

matplotlib.use('Agg')

m = 1.0
J = 0.01
g = 9.81
x_0 = 0.0
z_0 = 1.0
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

def controllability_matrix(A, B):
    n = A.shape[0]
    C = B.copy()
    A_power = np.eye(n)
    for i in range(1, n):
        A_power = A_power @ A
        C = np.hstack([C, A_power @ B])
    rank = np.linalg.matrix_rank(C)
    return C, rank, rank == n

def observability_matrix(A, C):
    n = A.shape[0]
    O = C.copy()
    A_power = np.eye(n)
    for i in range(1, n):
        A_power = A_power @ A
        O = np.vstack([O, C @ A_power])
    rank = np.linalg.matrix_rank(O)
    return O, rank, rank == n

def analyze_observability(A):
    n = A.shape[0]
    scenarios = {}
    
    C_full = np.eye(n)
    _, rank, obs = observability_matrix(A, C_full)
    scenarios['full'] = {'rank': rank, 'obs': obs}
    
    C_pos = np.array([[1,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,0]])
    _, rank, obs = observability_matrix(A, C_pos)
    scenarios['pos'] = {'rank': rank, 'obs': obs}
    
    C_vel = np.array([[0,1,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,1]])
    _, rank, obs = observability_matrix(A, C_vel)
    scenarios['vel'] = {'rank': rank, 'obs': obs}
    
    C_h = np.array([[0,0,1,0,0,0], [0,0,0,0,1,0]])
    _, rank, obs = observability_matrix(A, C_h)
    scenarios['height'] = {'rank': rank, 'obs': obs}
    
    return scenarios


def main():
    print("АНАЛИЗ МАТЕМАТИЧЕСКОЙ МОДЕЛИ")
    
    A, B = build_matrices()
    
    print("\n1. Параметры:")
    print(f"   m = {m} кг, J = {J} кг·м², g = {g} м/с²")
    print(f"   Точка равновесия: x₀={x_0}, z₀={z_0}, T₀={T_0:.2f} Н")
    
    print("\n2. Матрицы системы:")
    print("   A:")
    print(A)
    print("   B:")
    print(B)
    
    print("\n3. Управляемость:")
    C, rank, controllable = controllability_matrix(A, B)
    print(f"   Ранг матрицы управляемости: {rank}")
    print(f"   Система управляема: {'Да' if controllable else 'Нет'}")
    
    print("\n4. Наблюдаемость:")
    obs_scenarios = analyze_observability(A)
    for name, sc in obs_scenarios.items():
        print(f"   {name}: ранг={sc['rank']}, наблюдаема={sc['obs']}")
    
    print("\n5. Устойчивость:")
    eigenvalues = np.linalg.eigvals(A)
    print(f"   Собственные значения: {eigenvalues}")
    num_zero = np.sum(np.abs(eigenvalues) < 1e-10)
    print(f"   Нулевых собственных значений: {num_zero}")
    print("   Система на границе устойчивости")

    
    print("\nАнализ завершен")

if __name__ == "__main__":
    main()
