import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import ode
from lqr_design import build_matrices, design_lqr

m = 1.0
J = 0.01
g = 9.81
T_0 = m * g

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'mathtext.default': 'regular'
})

def quadcopter_dynamics(t, state, u_func):
    x, vx, z, vz, theta, omega = state
    T, M = u_func(t, state)
    
    dxdt = vx
    dvxdt = -(T / m) * np.sin(theta)
    dzdt = vz
    dvzdt = (T / m) * np.cos(theta) - g
    dthetadt = omega
    domegadt = M / J
    
    return [dxdt, dvxdt, dzdt, dvzdt, dthetadt, domegadt]

def simulate_lqr(x_ref, z_ref, x0, z0, theta0=0.0, omega0=0.0, 
                 dt=0.01, t_end=15.0, Q=None, R=None):
    A, B = build_matrices()
    K, _ = design_lqr(A, B, Q, R)
    
    state0 = [x0, 0.0, z0, 0.0, theta0, omega0]
    
    def control_law(t, state):
        x, vx, z, vz, theta, omega = state
        
        delta_x = x - x_ref
        delta_vx = vx
        delta_z = z - z_ref
        delta_vz = vz
        delta_theta = theta
        delta_omega = omega
        
        delta_state = np.array([delta_x, delta_vx, delta_z, delta_vz, delta_theta, delta_omega])
        delta_u = -K @ delta_state
        
        delta_T = delta_u[0]
        delta_M = delta_u[1]
        
        T = np.clip(T_0 + delta_T, 0, 2.0 * m * g)
        M = np.clip(delta_M, -0.8, 0.8)
        
        return [T, M]
    
    solver = ode(quadcopter_dynamics)
    solver.set_integrator('dopri5', rtol=1e-6, atol=1e-9)
    solver.set_initial_value(state0, 0.0)
    solver.set_f_params(control_law)
    
    t = [0.0]
    states = [state0.copy()]
    controls = [control_law(0.0, state0)]
    
    while solver.t < t_end:
        solver.integrate(solver.t + dt)
        if not solver.successful():
            break
        t.append(solver.t)
        state = solver.y
        states.append(state.copy())
        controls.append(control_law(solver.t, state))
    
    t = np.array(t)
    states = np.array(states)
    controls = np.array(controls)
    
    x_hist = states[:, 0]
    vx_hist = states[:, 1]
    z_hist = states[:, 2]
    vz_hist = states[:, 3]
    theta_hist = states[:, 4]
    omega_hist = states[:, 5]
    T_hist = controls[:, 0]
    M_hist = controls[:, 1]
    
    return t, x_hist, z_hist, theta_hist, T_hist, M_hist, vx_hist, vz_hist, omega_hist

def evaluate_performance(t, x, z, theta, x_ref, z_ref):
    x_err = x - x_ref
    z_err = z - z_ref
    theta_err = theta
    
    settling_tolerance = 0.05
    
    def find_settling_time(err, tolerance):
        target = np.abs(err[-1])
        threshold = max(settling_tolerance * np.abs(err[-1] + err[0] - target), 0.01)
        for i in range(len(err) - 1, -1, -1):
            if np.abs(err[i] - err[-1]) > threshold:
                if i < len(err) - 1:
                    return t[i + 1]
                else:
                    return t[-1]
        return t[0]
    
    def find_overshoot(values, target):
        if len(values) == 0:
            return 0.0
        final_val = values[-1]
        if np.abs(target - final_val) < 1e-6:
            max_dev = np.max(np.abs(values - final_val))
            return (max_dev / np.abs(final_val - values[0])) * 100 if np.abs(final_val - values[0]) > 1e-6 else 0.0
        max_val = np.max(values) if target > final_val else np.min(values)
        max_dev = np.abs(max_val - final_val)
        initial_dev = np.abs(values[0] - final_val)
        return (max_dev / initial_dev * 100) if initial_dev > 1e-6 else 0.0
    
    def find_steady_state_error(err):
        final_indices = int(len(err) * 0.2)
        if final_indices == 0:
            final_indices = 1
        return np.abs(np.mean(err[-final_indices:]))
    
    ts_x = find_settling_time(x_err, settling_tolerance)
    ts_z = find_settling_time(z_err, settling_tolerance)
    ts_theta = find_settling_time(theta_err, settling_tolerance)
    
    overshoot_x = find_overshoot(x, x_ref)
    overshoot_z = find_overshoot(z, z_ref)
    overshoot_theta = find_overshoot(theta, 0.0)
    
    sse_x = find_steady_state_error(x_err)
    sse_z = find_steady_state_error(z_err)
    sse_theta = find_steady_state_error(theta_err)
    
    return {
        'ts_x': ts_x, 'ts_z': ts_z, 'ts_theta': ts_theta,
        'overshoot_x': overshoot_x, 'overshoot_z': overshoot_z, 'overshoot_theta': overshoot_theta,
        'sse_x': sse_x, 'sse_z': sse_z, 'sse_theta': sse_theta
    }

def plot_results(t, x, z, theta, T, M, x_ref, z_ref, save_dir='plots'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # График горизонтального положения x(t)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, x, 'b-', linewidth=2, label=r'$x(t)$')
    ax.axhline(y=x_ref, color='r', linestyle='--', linewidth=2, label=r'$x_{\mathrm{ref}}$')
    ax.set_xlabel(r'$t$, с', fontsize=14)
    ax.set_ylabel(r'$x$, м', fontsize=14)
    ax.set_title(r'Горизонтальное положение $x(t)$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/x_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # График высоты z(t)
    z_plot = np.maximum(z, 0)  # Ограничение: высота не может быть отрицательной
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, z_plot, 'b-', linewidth=2, label=r'$z(t)$')
    ax.axhline(y=z_ref, color='r', linestyle='--', linewidth=2, label=r'$z_{\mathrm{ref}}$')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5, label='Земля')
    ax.set_xlabel(r'$t$, с', fontsize=14)
    ax.set_ylabel(r'$z$, м', fontsize=14)
    ax.set_title(r'Высота $z(t)$', fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/z_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # График ориентации theta(t)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, np.degrees(theta), 'b-', linewidth=2, label=r'$\theta(t)$')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label=r'$\theta_{\mathrm{ref}}$')
    ax.set_xlabel(r'$t$, с', fontsize=14)
    ax.set_ylabel(r'$\theta$, град', fontsize=14)
    ax.set_title(r'Ориентация $\theta(t)$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/theta_orientation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # График тяги T(t)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, T, 'g-', linewidth=2)
    ax.axhline(y=T_0, color='r', linestyle='--', linewidth=1, label=r'$T_0 = mg$')
    ax.set_xlabel(r'$t$, с', fontsize=14)
    ax.set_ylabel(r'$T$, Н', fontsize=14)
    ax.set_title(r'Управляющее воздействие: тяга $T(t)$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/thrust_T.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # График момента M(t)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, M, 'm-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$t$, с', fontsize=14)
    ax.set_ylabel(r'$M$, Н$\cdot$м', fontsize=14)
    ax.set_title(r'Управляющее воздействие: момент $M(t)$', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/moment_M.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в директории {save_dir}/")

if __name__ == '__main__':
    x_ref = 2.0
    z_ref = 1.5
    x0 = 0.0
    z0 = 1.0
    theta0 = 0.1
    
    Q = np.diag([100, 10, 100, 10, 50, 5])
    R = np.diag([1, 0.01])
    
    print("Запуск моделирования с LQR-регулятором...")
    t, x, z, theta, T, M, vx, vz, omega = simulate_lqr(
        x_ref, z_ref, x0, z0, theta0, dt=0.01, t_end=10.0, Q=Q, R=R
    )
    
    print("\nРезультаты моделирования:")
    print(f"x_final = {x[-1]:.4f} м (x_ref = {x_ref} м)")
    print(f"z_final = {z[-1]:.4f} м (z_ref = {z_ref} м)")
    print(f"theta_final = {np.degrees(theta[-1]):.4f}° (theta_ref = 0°)")
    
    perf = evaluate_performance(t, x, z, theta, x_ref, z_ref)
    print("\nПоказатели качества:")
    print(f"Время установления x: {perf['ts_x']:.2f} с")
    print(f"Время установления z: {perf['ts_z']:.2f} с")
    print(f"Время установления θ: {perf['ts_theta']:.2f} с")
    print(f"Перерегулирование x: {perf['overshoot_x']:.2f}%")
    print(f"Перерегулирование z: {perf['overshoot_z']:.2f}%")
    print(f"Перерегулирование θ: {perf['overshoot_theta']:.2f}%")
    print(f"Статическая ошибка x: {perf['sse_x']:.4f} м")
    print(f"Статическая ошибка z: {perf['sse_z']:.4f} м")
    print(f"Статическая ошибка θ: {perf['sse_theta']:.4f} рад")
    
    plot_results(t, x, z, theta, T, M, x_ref, z_ref, 'plots')
