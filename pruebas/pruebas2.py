import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros del Ejercicio ---
U_ef = 120  # Tensión eficaz de fase (Vrms)
R_c = 5     # Resistencia de la carga (Ohms)
alpha_deg = 90  # Ángulo de disparo (grados)
alpha = np.deg2rad(alpha_deg)  # Ángulo de disparo (radianes)

# Cálculo de la Tensión Máxima de Pico (Vm)
Vm = U_ef * np.sqrt(2)
# Vm ≈ 169.71 V

# --- Simulación ---
omega_t = np.linspace(0, 2 * np.pi, 500)  # Un ciclo completo (0 a 2*pi)

# 1. Tensión de las fuentes (u1 y u2)
u1 = Vm * np.sin(omega_t)
u2 = Vm * np.sin(omega_t - np.pi) # u2 = -u1

# 2. Tensión de carga (vc) para carga puramente resistiva
vc = np.zeros_like(omega_t)

for i, t in enumerate(omega_t):
    # Condición de Conducción para Rectificador Bifásico Controlado (Carga R)
    # Tiristor X1 conduce (vc = u1) de alpha a pi
    if t >= alpha and t < np.pi:
        vc[i] = u1[i]
    # Tiristor X2 conduce (vc = u2) de pi + alpha a 2*pi
    elif t >= np.pi + alpha and t < 2 * np.pi:
        vc[i] = u2[i]
    else:
        # Bloqueo (vc = 0)
        vc[i] = 0

# 3. Corriente de Carga (ic)
ic = vc / R_c

# --- Gráfica ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f'Rectificador Controlado Bifásico (Punto Medio) | $\\alpha={alpha_deg}^\circ$')

# --- Gráfico de Tensiones (vc y Fuentes) ---
ax1.plot(omega_t, u1, 'k--', alpha=0.4, label='$u_1$')
ax1.plot(omega_t, u2, 'k:', alpha=0.4, label='$u_2$')
ax1.plot(omega_t, vc, 'r', linewidth=2, label='$v_c$ (Carga)')

# Etiquetas de tensión
ax1.set_ylabel('Tensión [V]')
ax1.set_title(f'Tensión de Pico $V_m \\approx {Vm:.2f}$ V')

# Marcar el ángulo alfa
ax1.plot([alpha, alpha], [0, Vm], 'b--', alpha=0.7)
ax1.plot([np.pi + alpha, np.pi + alpha], [0, Vm], 'b--', alpha=0.7, label='$\\alpha$')
ax1.text(alpha, -Vm*0.1, '$\\alpha$', color='blue', ha='center')
ax1.text(np.pi + alpha, -Vm*0.1, '$\\pi+\\alpha$', color='blue', ha='center')

ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')
ax1.set_ylim(-Vm*0.15, Vm*1.15)

# --- Gráfico de Corriente (ic) ---
I_max = Vm / R_c # Corriente máxima teórica (≈ 33.94 A)

ax2.plot(omega_t, ic, 'orange', linewidth=2, label='$i_c$ (Carga)')

# Cálculos de valor medio para etiquetado
Vcd = (Vm / np.pi) * (1 + np.cos(alpha))
Icd = Vcd / R_c

ax2.set_ylabel('Corriente [A]')
ax2.set_xlabel('Ángulo ($\omega t$) [rad]')
ax2.set_title(f'Corriente Media $I_{{cd}} \\approx {Icd:.2f}$ A | Corriente Máx $I_{{max}} \\approx {I_max:.2f}$ A')

# Configurar el eje X para mostrar valores de pi
pi_ticks = np.arange(0, 2.1 * np.pi, np.pi / 4)
pi_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']
ax2.set_xticks(pi_ticks)
ax2.set_xticklabels(pi_labels[:len(pi_ticks)])
ax2.set_xlim(0, 2 * np.pi)

ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')
ax2.set_ylim(-I_max*0.1, I_max*1.15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()