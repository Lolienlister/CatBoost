"""
Зонные диаграммы p-n-переходов (задание 1).

Три перехода:
  1) n-Si (Nd=1e18) | p-Si (Na=1e15)
  2) p-Si (Na=1e19) | n-Si (Nd=1e14)
  3) n-Si (Nd=1e17) | p-Si (Na=1e12)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
})

out_dir = "/home/ubuntu/CatBoost/graphs"
os.makedirs(out_dir, exist_ok=True)

# ── Физические константы Si при T = 300 K ────────────────────
Eg   = 1.12       # eV, ширина запрещённой зоны
kT   = 0.026      # eV
ni   = 1.5e10      # cm^-3, собственная концентрация
chi  = 4.05       # eV, электронное сродство


def calc_params(Nd, Na):
    """Рассчитать параметры зонной диаграммы p-n-перехода."""
    # Положение уровня Ферми относительно середины запрещённой зоны
    phi_n = kT * np.log(Nd / ni)   # Ef - Ei на n-стороне
    phi_p = kT * np.log(Na / ni)   # Ei - Ef на p-стороне

    # Расстояния от зонных краёв до Ef
    Ec_minus_Ef_n = Eg / 2.0 - phi_n   # Ec - Ef на n-стороне
    Ef_minus_Ev_p = Eg / 2.0 - phi_p   # Ef - Ev на p-стороне

    # Контактная разность потенциалов
    V_bi = kT * np.log(Nd * Na / ni**2)

    # Отношение ширин ОПЗ: xn/xp = Na/Nd
    ratio = Na / Nd   # xn / xp

    return {
        'phi_n': phi_n,
        'phi_p': phi_p,
        'Ec_Ef_n': Ec_minus_Ef_n,
        'Ef_Ev_p': Ef_minus_Ev_p,
        'V_bi': V_bi,
        'xn_over_xp': ratio,
    }


def draw_band_diagram(ax, Nd, Na, title, n_left=True):
    """
    Нарисовать зонную диаграмму на оси ax.

    n_left=True  → n слева, p справа  (переходы 1 и 3)
    n_left=False → p слева, n справа  (переход 2)
    """
    p = calc_params(Nd, Na)
    V_bi = p['V_bi']

    # --- Координаты по x -------------------------------------------------
    # Общая длина = 10 условных единиц
    # ОПЗ в центре; делим ОПЗ пропорционально
    W_total = 3.0  # условная ширина ОПЗ на графике
    if n_left:
        frac_n = p['xn_over_xp'] / (1.0 + p['xn_over_xp'])
    else:
        frac_n = 1.0 / (1.0 + p['xn_over_xp'])

    frac_n = np.clip(frac_n, 0.05, 0.95)
    W_n = W_total * frac_n
    W_p = W_total * (1.0 - frac_n)

    x_junction = 5.0
    if n_left:
        x_dep_left  = x_junction - W_n
        x_dep_right = x_junction + W_p
    else:
        x_dep_left  = x_junction - W_p
        x_dep_right = x_junction + W_n

    # --- Энергии в плоских участках (далеко от перехода) -------------------
    # Примем Ef = 0 (уровень Ферми — ось отсчёта)
    Ef = 0.0

    # n-сторона
    Ec_n = Ef + p['Ec_Ef_n']
    Ev_n = Ec_n - Eg
    Ei_n = (Ec_n + Ev_n) / 2.0

    # p-сторона
    Ev_p = Ef - p['Ef_Ev_p']
    Ec_p = Ev_p + Eg
    Ei_p = (Ec_p + Ev_p) / 2.0

    # --- Построение кривых ------------------------------------------------
    N_pts = 200

    # Плоские участки
    x_flat_left  = np.linspace(0, x_dep_left, 40)
    x_flat_right = np.linspace(x_dep_right, 10, 40)

    # ОПЗ — параболический изгиб зон
    x_dep = np.linspace(x_dep_left, x_dep_right, N_pts)

    if n_left:
        Ec_left, Ev_left, Ei_left = Ec_n, Ev_n, Ei_n
        Ec_right, Ev_right, Ei_right = Ec_p, Ev_p, Ei_p
    else:
        Ec_left, Ev_left, Ei_left = Ec_p, Ev_p, Ei_p
        Ec_right, Ev_right, Ei_right = Ec_n, Ev_n, Ei_n

    # Интерполяция зон в ОПЗ (сплайн-подобная S-кривая)
    t = (x_dep - x_dep_left) / (x_dep_right - x_dep_left)
    s = 3 * t**2 - 2 * t**3  # гладкий переход (Hermite)

    Ec_dep = Ec_left + (Ec_right - Ec_left) * s
    Ev_dep = Ev_left + (Ev_right - Ev_left) * s
    Ei_dep = Ei_left + (Ei_right - Ei_left) * s

    # Полные массивы
    x_all  = np.concatenate([x_flat_left, x_dep, x_flat_right])
    Ec_all = np.concatenate([np.full_like(x_flat_left, Ec_left),
                             Ec_dep,
                             np.full_like(x_flat_right, Ec_right)])
    Ev_all = np.concatenate([np.full_like(x_flat_left, Ev_left),
                             Ev_dep,
                             np.full_like(x_flat_right, Ev_right)])
    Ei_all = np.concatenate([np.full_like(x_flat_left, Ei_left),
                             Ei_dep,
                             np.full_like(x_flat_right, Ei_right)])

    # --- Рисуем -----------------------------------------------------------
    ax.plot(x_all, Ec_all, 'b-',  linewidth=2.2, label='$E_c$')
    ax.plot(x_all, Ev_all, 'r-',  linewidth=2.2, label='$E_v$')
    ax.plot(x_all, Ei_all, 'g--', linewidth=1.4, label='$E_i$')
    ax.axhline(Ef, color='k', linestyle='-.', linewidth=1.6, label='$E_F$')

    # Штриховка ОПЗ
    y_min = min(Ev_all.min(), Ev_left, Ev_right) - 0.15
    y_max = max(Ec_all.max(), Ec_left, Ec_right) + 0.15
    ax.axvspan(x_dep_left, x_dep_right, alpha=0.07, color='gray')

    # Вертикальная пунктирная линия — металлургический контакт
    ax.axvline(x_junction, color='gray', linestyle=':', linewidth=1.0)

    # Подписи сторон
    if n_left:
        ax.text(1.5, y_max - 0.08, f'n-Si\n$N_D$ = {Nd:.0e} см⁻³',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='#dce6f7', alpha=0.8))
        ax.text(8.5, y_max - 0.08, f'p-Si\n$N_A$ = {Na:.0e} см⁻³',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='#f7dcdc', alpha=0.8))
    else:
        ax.text(1.5, y_max - 0.08, f'p-Si\n$N_A$ = {Na:.0e} см⁻³',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='#f7dcdc', alpha=0.8))
        ax.text(8.5, y_max - 0.08, f'n-Si\n$N_D$ = {Nd:.0e} см⁻³',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='#dce6f7', alpha=0.8))

    # Аннотация qV_bi
    arr_x = x_junction
    ax.annotate('', xy=(arr_x + 0.3, Ec_right), xytext=(arr_x + 0.3, Ec_left),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    mid_y = (Ec_left + Ec_right) / 2.0
    ax.text(arr_x + 0.6, mid_y,
            f'$qV_{{bi}}$ = {V_bi:.3f} эВ', fontsize=11, color='purple', va='center')

    # Подписи зон справа
    ax.text(10.15, Ec_right, '$E_c$', fontsize=12, color='blue', va='center')
    ax.text(10.15, Ev_right, '$E_v$', fontsize=12, color='red', va='center')
    ax.text(10.15, Ei_right, '$E_i$', fontsize=12, color='green', va='center')
    ax.text(10.15, Ef, '$E_F$', fontsize=12, color='black', va='center')

    ax.text(5.0, y_min + 0.05, 'ОПЗ', ha='center', fontsize=10, color='gray')

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(y_min, y_max + 0.15)
    ax.set_xlabel('x (усл. ед.)', fontsize=12)
    ax.set_ylabel('E, эВ', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)


# ══════════════════════════════════════════════════════════════
# Диаграмма 1: n-Si (Nd=1e18) | p-Si (Na=1e15)
# ══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 6))
draw_band_diagram(ax1, Nd=1e18, Na=1e15,
                  title='Зонная диаграмма p-n-перехода: n-Si ($N_D$=10$^{18}$) | p-Si ($N_A$=10$^{15}$)',
                  n_left=True)
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, 'band_diagram_1.png'), dpi=200)
plt.close(fig1)
print("1) band_diagram_1.png saved")

# ══════════════════════════════════════════════════════════════
# Диаграмма 2: p-Si (Na=1e19) | n-Si (Nd=1e14)
# ══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 6))
draw_band_diagram(ax2, Nd=1e14, Na=1e19,
                  title='Зонная диаграмма p-n-перехода: p-Si ($N_A$=10$^{19}$) | n-Si ($N_D$=10$^{14}$)',
                  n_left=False)
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'band_diagram_2.png'), dpi=200)
plt.close(fig2)
print("2) band_diagram_2.png saved")

# ══════════════════════════════════════════════════════════════
# Диаграмма 3: n-Si (Nd=1e17) | p-Si (Na=1e12)
# ══════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 6))
draw_band_diagram(ax3, Nd=1e17, Na=1e12,
                  title='Зонная диаграмма p-n-перехода: n-Si ($N_D$=10$^{17}$) | p-Si ($N_A$=10$^{12}$)',
                  n_left=True)
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, 'band_diagram_3.png'), dpi=200)
plt.close(fig3)
print("3) band_diagram_3.png saved")

print("\nAll band diagrams generated in", out_dir)
