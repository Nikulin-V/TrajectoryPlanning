import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import ode

matplotlib.use('Agg')

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

def quadcopter_dynamics_free(t, state):
    x, vx, z, vz, theta, omega = state
    
    dxdt = vx
    dvxdt = -(T_0 / m) * np.sin(theta)
    dzdt = vz
    dvzdt = (T_0 / m) * np.cos(theta) - g
    dthetadt = omega
    domegadt = 0.0
    
    return [dxdt, dvxdt, dzdt, dvzdt, dthetadt, domegadt]

def simulate_free_motion(x0, z0, vx0, vz0, theta0, omega0, dt=0.01, t_end=5.0):
    state0 = [x0, vx0, z0, vz0, theta0, omega0]
    
    solver = ode(quadcopter_dynamics_free)
    solver.set_integrator('dopri5', rtol=1e-6, atol=1e-9)
    solver.set_initial_value(state0, 0.0)
    
    t = [0.0]
    states = [state0.copy()]
    
    while solver.t < t_end:
        solver.integrate(solver.t + dt)
        if not solver.successful() or solver.y[2] < 0:
            break
        t.append(solver.t)
        states.append(solver.y.copy())
    
    t = np.array(t)
    states = np.array(states)
    
    x_hist = states[:, 0]
    vx_hist = states[:, 1]
    z_hist = states[:, 2]
    vz_hist = states[:, 3]
    theta_hist = states[:, 4]
    omega_hist = states[:, 5]
    
    return t, x_hist, z_hist, theta_hist, vx_hist, vz_hist, omega_hist

def simulate_with_disturbances(x_ref, z_ref, x0, z0, theta0, dist_type, dist_time, dist_value, 
                               Q=None, R=None, dt=0.01, t_end=10.0):
    from lqr_design import build_matrices, design_lqr
    
    A, B = build_matrices()
    K, _ = design_lqr(A, B, Q, R)
    
    state0 = [x0, 0.0, z0, 0.0, theta0, 0.0]
    
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
        
        if dist_type == 'thrust' and abs(t - dist_time) < dt/2:
            delta_T += dist_value
        elif dist_type == 'moment' and dist_time <= t < dist_time + 0.2:
            delta_M += dist_value
        
        T = np.clip(T_0 + delta_T, 0, 2.0 * m * g)
        M = np.clip(delta_M, -0.8, 0.8)
        
        return [T, M]
    
    def quadcopter_dynamics_controlled(t, state, u_func):
        x, vx, z, vz, theta, omega = state
        
        # Ограничение: высота не может быть отрицательной
        if z < 0:
            z = 0
            vz = max(0, vz)  # Скорость вниз становится нулевой при касании земли
        
        T, M = u_func(t, state)
        
        wind_force = 0.0
        if dist_type == 'wind' and dist_time <= t < dist_time + 0.5:
            wind_force = dist_value * m
        
        dxdt = vx
        dvxdt = -(T / m) * np.sin(theta) - wind_force / m
        dzdt = vz if z > 0 else 0  # Остановка при z=0
        dvzdt = (T / m) * np.cos(theta) - g if z > 0 else 0
        dthetadt = omega
        domegadt = M / J
        
        return [dxdt, dvxdt, dzdt, dvzdt, dthetadt, domegadt]
    
    solver = ode(quadcopter_dynamics_controlled)
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
        state = solver.y
        # Остановка симуляции, если квадрокоптер упал
        if state[2] < 0:
            state[2] = 0
            state[3] = 0
            break
        t.append(solver.t)
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
    
    return t, x_hist, z_hist, theta_hist, T_hist, M_hist

def plot_free_motion(t, x, z, theta, save_dir='plots'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(t, x, 'b-', linewidth=2)
    axes[0].set_xlabel(r'$t$, с', fontsize=14)
    axes[0].set_ylabel(r'$x$, м', fontsize=14)
    axes[0].set_title(r'Свободное движение: $x(t)$', fontsize=14)
    axes[0].grid(True)
    
    axes[1].plot(t, z, 'b-', linewidth=2)
    axes[1].set_xlabel(r'$t$, с', fontsize=14)
    axes[1].set_ylabel(r'$z$, м', fontsize=14)
    axes[1].set_title(r'Свободное движение: $z(t)$', fontsize=14)
    axes[1].grid(True)
    
    axes[2].plot(t, np.degrees(theta), 'b-', linewidth=2)
    axes[2].set_xlabel(r'$t$, с', fontsize=14)
    axes[2].set_ylabel(r'$\theta$, град', fontsize=14)
    axes[2].set_title(r'Свободное движение: $\theta(t)$', fontsize=14)
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/free_motion.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_disturbances(t, x, z, theta, T, M, x_ref, z_ref, dist_type, save_dir='plots'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Ограничение: высота не может быть отрицательной
    z_plot = np.maximum(z, 0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Реакция системы на возмущение типа: {dist_type}', fontsize=14)
    
    axes[0, 0].plot(t, x, 'b-', linewidth=2, label=r'$x(t)$')
    axes[0, 0].axhline(y=x_ref, color='r', linestyle='--', linewidth=2, label=r'$x_{\mathrm{ref}}$')
    axes[0, 0].set_xlabel(r'$t$, с', fontsize=12)
    axes[0, 0].set_ylabel(r'$x$, м', fontsize=12)
    axes[0, 0].set_title(r'Горизонтальное положение', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, z_plot, 'b-', linewidth=2, label=r'$z(t)$')
    axes[0, 1].axhline(y=z_ref, color='r', linestyle='--', linewidth=2, label=r'$z_{\mathrm{ref}}$')
    axes[0, 1].axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5, label='Земля')
    axes[0, 1].set_xlabel(r'$t$, с', fontsize=12)
    axes[0, 1].set_ylabel(r'$z$, м', fontsize=12)
    axes[0, 1].set_title(r'Высота', fontsize=12)
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(t, np.degrees(theta), 'b-', linewidth=2)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 2].set_xlabel(r'$t$, с', fontsize=12)
    axes[0, 2].set_ylabel(r'$\theta$, град', fontsize=12)
    axes[0, 2].set_title(r'Ориентация', fontsize=12)
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(t, T, 'g-', linewidth=2)
    axes[1, 0].axhline(y=T_0, color='r', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel(r'$t$, с', fontsize=12)
    axes[1, 0].set_ylabel(r'$T$, Н', fontsize=12)
    axes[1, 0].set_title(r'Тяга', fontsize=12)
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, M, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel(r'$t$, с', fontsize=12)
    axes[1, 1].set_ylabel(r'$M$, Н$\cdot$м', fontsize=12)
    axes[1, 1].set_title(r'Момент', fontsize=12)
    axes[1, 1].grid(True)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/disturbance_{dist_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("1. Моделирование свободного движения...")
    t_free, x_free, z_free, theta_free, _, _, _ = simulate_free_motion(
        x0=0.0, z0=1.0, vx0=0.0, vz0=0.0, theta0=0.1, omega0=0.0, dt=0.01, t_end=3.0
    )
    plot_free_motion(t_free, x_free, z_free, theta_free, 'plots')
    print(f"   Время падения: {t_free[-1]:.2f} с")
    print(f"   Финальная высота: {z_free[-1]:.4f} м")
    
    print("\n2. Моделирование с различными начальными условиями...")
    Q = np.diag([100, 10, 100, 10, 50, 5])
    R = np.diag([1, 0.01])
    
    scenarios = [
        {'name': 'large_theta', 'x0': 0.0, 'z0': 1.0, 'theta0': 0.3, 'x_ref': 0.0, 'z_ref': 1.0},
        {'name': 'large_x_dev', 'x0': 1.0, 'z0': 1.0, 'theta0': 0.0, 'x_ref': 0.0, 'z_ref': 1.0},
        {'name': 'large_z_dev', 'x0': 0.0, 'z0': 0.5, 'theta0': 0.0, 'x_ref': 0.0, 'z_ref': 1.0},
    ]
    
    for scenario in scenarios:
        t, x, z, theta, T, M = simulate_with_disturbances(
            scenario['x_ref'], scenario['z_ref'], 
            scenario['x0'], scenario['z0'], scenario['theta0'],
            'none', 0, 0, Q, R, dt=0.01, t_end=8.0
        )
        plot_disturbances(t, x, z, theta, T, M, scenario['x_ref'], scenario['z_ref'], 
                         f"initial_{scenario['name']}", 'plots')
        print(f"   Сценарий {scenario['name']}: x_final={x[-1]:.3f} м, z_final={z[-1]:.3f} м")
    
    print("\n3. Моделирование с возмущениями...")
    disturbances = [
        {'type': 'thrust', 'time': 2.0, 'value': 2.0},
        {'type': 'wind', 'time': 2.0, 'value': 1.0},
    ]
    
    for dist in disturbances:
        t, x, z, theta, T, M = simulate_with_disturbances(
            0.0, 1.0, 0.0, 1.0, 0.0,
            dist['type'], dist['time'], dist['value'], Q, R, dt=0.01, t_end=8.0
        )
        plot_disturbances(t, x, z, theta, T, M, 0.0, 1.0, dist['type'], 'plots')
        print(f"   Возмущение {dist['type']}: максимальное отклонение x={np.max(np.abs(x-0.0)):.3f} м, z={np.max(np.abs(z-1.0)):.3f} м")
    
    print("\nГрафики сохранены в директории plots/")

