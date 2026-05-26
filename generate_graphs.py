import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

out_dir = "/home/ubuntu/CatBoost/graphs"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Chart 1: ВАХ прямой ток первого диода
# ============================================================
U1_fwd = [0.075, 0.15, 0.225, 0.3, 0.375, 0.45, 0.525, 0.6]
I1_fwd = [0.1, 1.5, 5.0, 11.2, 23.0, 37.5, 58.0, 69.2]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(U1_fwd, I1_fwd, 'o-', color='#1f77b4', markersize=7, linewidth=2, label='Диод 1 (прямой ток)')
ax.set_xlabel('U, В', fontsize=13)
ax.set_ylabel('I, мА', fontsize=13)
ax.set_title('ВАХ первого диода при прямом токе', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig1_vah_forward_diode1.png'), dpi=200)
plt.close(fig)

# ============================================================
# Chart 2: ВАХ обратный ток первого диода
# ============================================================
U1_rev = [5, 10, 15, 20, 25, 30, 35, 40]
I1_rev = [2.7, 3.4, 4.0, 4.7, 5.4, 6.3, 7.2, 8.5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(U1_rev, I1_rev, 's-', color='#d62728', markersize=7, linewidth=2, label='Диод 1 (обратный ток)')
ax.set_xlabel('U, В', fontsize=13)
ax.set_ylabel('I, мА', fontsize=13)
ax.set_title('ВАХ первого диода при обратном токе', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig2_vah_reverse_diode1.png'), dpi=200)
plt.close(fig)

# ============================================================
# Chart 3: ВАХ для второго диода
# ============================================================
U2 = [0.075, 0.15, 0.225, 0.3, 0.375, 0.45, 0.525, 0.6, 0.675]
I2 = [0.1, 0.1, 0.1, 0.2, 1.0, 5.0, 13.9, 31.8, 51.5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(U2, I2, '^-', color='#2ca02c', markersize=7, linewidth=2, label='Диод 2 (прямой ток)')
ax.set_xlabel('U, В', fontsize=13)
ax.set_ylabel('I, мА', fontsize=13)
ax.set_title('ВАХ для второго диода', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig3_vah_forward_diode2.png'), dpi=200)
plt.close(fig)

# ============================================================
# Chart 4: Зависимость I от T
# ============================================================
T_vals = [28, 30, 35, 40, 45, 50, 55, 60, 65, 70]
I_sat  = [6.8, 7.4, 10.1, 15.2, 21.6, 31.0, 43.8, 60.0, 83.8, 115.0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(T_vals, I_sat, 'D-', color='#9467bd', markersize=7, linewidth=2, label='I$_{нас}$(T)')
ax.set_xlabel('T, °C', fontsize=13)
ax.set_ylabel('I$_{нас}$, мкА', fontsize=13)
ax.set_title('Зависимость обратного тока насыщения от температуры', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_i_vs_t.png'), dpi=200)
plt.close(fig)

# ============================================================
# Chart 5: ln(Is) vs 1000/T with linear trendline
# ============================================================
inv_T = [3.322, 3.3, 3.247, 3.195, 3.145, 3.096, 3.049, 3.003, 2.959, 2.915]
ln_I  = [1.92, 2.0, 2.31, 2.72, 3.07, 3.43, 3.78, 4.09, 4.43, 4.75]

coeffs = np.polyfit(inv_T, ln_I, 1)
poly = np.poly1d(coeffs)
x_fit = np.linspace(min(inv_T) - 0.05, max(inv_T) + 0.05, 100)
y_fit = poly(x_fit)

r_squared = 1 - np.sum((np.array(ln_I) - poly(np.array(inv_T)))**2) / np.sum((np.array(ln_I) - np.mean(ln_I))**2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(inv_T, ln_I, color='#ff7f0e', s=60, zorder=5, label='Экспериментальные данные')
ax.plot(x_fit, y_fit, '--', color='#1f77b4', linewidth=2,
        label=f'Аппроксимация: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}\nR² = {r_squared:.4f}')
ax.set_xlabel('1000/T, K$^{-1}$', fontsize=13)
ax.set_ylabel('ln(I$_s$)', fontsize=13)
ax.set_title('Зависимость ln(I$_s$) от 1000/T', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_ln_i_vs_inv_t.png'), dpi=200)
plt.close(fig)

print("All 5 graphs generated in", out_dir)
for f in sorted(os.listdir(out_dir)):
    print(f"  {f}")
