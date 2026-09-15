# Частина A: приймальна перевірка ядра + підготовка даних для частини B.
# Запуск: python lab11/part_a_check.py

import os
import numpy as np

import arma_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TEST = os.path.join(HERE, "..", "lab_task", "ATS_Lab_01_new", "ATS_Lab_01_new", "Test+")

N = 1000
SEED = 11
np.set_printoptions(precision=6, suppress=True)


def head(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------- 1. модель та її стійкість
head("1. Модель варіанта 11 та перевірка стійкості")
a, b = core.A_TRUE, core.B_TRUE
sgn = lambda x: ("+ %g" % x) if x >= 0 else ("- %g" % abs(x))
print("y(k) = %g %s*y(k-1) %s*y(k-2) %s*y(k-3)" % (a[0], sgn(a[1]), sgn(a[2]), sgn(a[3])))
print("           + v(k) %s*v(k-1) %s*v(k-2) %s*v(k-3)" % (sgn(b[0]), sgn(b[1]), sgn(b[2])))
s_abs, s_raw, ok = core.check_stability()
print("sum|a_i| = %.3f  (< 1 -> %s);  sum a_i = %.3f" % (s_abs, ok, s_raw))

# ------------------------------------------------------------- 2. білий шум
head("2. Білий шум v(k)")
v = core.generate_noise(N, seed=SEED)
v12 = core.generate_noise_uniform12(N, seed=SEED)
print("normal():      N = %d, mean = %+.4f, var = %.4f" % (N, v.mean(), v.var()))
print("формула 12*U:  N = %d, mean = %+.4f, var = %.4f" % (N, v12.mean(), v12.var()))
print("корельованість сусідніх значень: r(1) = %+.4f" % np.corrcoef(v[:-1], v[1:])[0, 1])

# ---------------------------------------------------------- 3. еталонний ряд
head("3. Еталонний ряд y(k) за повною моделлю АРКС(3,3)")
y = core.generate_series(v)
print("N = %d, mean = %+.4f (теоретичне a0/(1-sum a_i) = %+.4f), std = %.4f"
      % (N, y.mean(), a[0] / (1 - s_raw), y.std()))
print("розбіжності немає:", bool(np.all(np.isfinite(y))) and abs(y).max() < 1e3,
      " max|y| = %.3f" % abs(y).max())

# ------------------------------------------- 4. тест на даних викладача
head("4. Тест на даних викладача (Test+): АРКС(3,1) з test.txt")
vt, yt = core.load_series(os.path.join(TEST, "v.txt")), core.load_series(os.path.join(TEST, "y.txt"))
vt, yt = core.align_loaded(vt, yt)
print("len(v) = %d, len(y) = %d -> ряд y починається з k = %d"
      % (len(vt), np.isfinite(yt).sum(), int(np.flatnonzero(np.isfinite(yt))[0])))

Xt, margin_t, Yt = core.generate_X(3, 1, vt, yt)
print("X: %s, перше рівняння k = %d" % (Xt.shape, margin_t))
th_t = core.mnk(Xt, Yt)
expected = np.array([0.005, 0.1, 0.2, 0.3, 0.005, 0.01])
print("МНК   theta = ", th_t)
print("очікуємо      ", expected, "   (a0..a3 і b1 з test.txt)")
err_t = float(np.abs(th_t - expected).max())
S_t = float(((Yt - Xt @ th_t) ** 2).sum())
print("max похибка = %.3e,  S = %.3e" % (err_t, S_t))
print("ТЕСТ ВИКЛАДАЧА:", "OK" if err_t < 1e-8 else "ПРОВАЛЕНО")

th_rt, _ = core.rmnk(Xt, Yt)
print("РМНК  theta = ", th_rt, " max похибка = %.3e" % float(np.abs(th_rt - expected).max()))

# --------------------------------- 5. самоперевірка на своєму варіанті (3,3)
head("5. Самоперевірка на варіанті 11, повний порядок АРКС(3,3)")
X, margin, Y = core.generate_X(3, 3, v, y)
print("X: %s, перше рівняння k = %d" % (X.shape, margin))
th_true = core.theta_true()
th_mnk = core.mnk(X, Y)
th_rmnk, thetas = core.rmnk(X, Y)
print("істинне theta = ", th_true)
print("МНК           = ", th_mnk, " max похибка = %.3e" % float(np.abs(th_mnk - th_true).max()))
print("РМНК          = ", th_rmnk, " max похибка = %.3e" % float(np.abs(th_rmnk - th_true).max()))

tol = 0.01
conv = np.abs(thetas - th_true).max(axis=1)
k_conv = int(np.argmax(conv < tol)) if np.any(conv < tol) else -1
print("РМНК увійшов у коридор +-%.2f на кроці k = %d (і далі max відхилення %.2e)"
      % (tol, k_conv, conv[k_conv:].max()))

# ------------------------------------------------------ 6. критерії якості
head("6. Критерії S, R^2, IKA (перевірка формул)")
for (p, q) in [(3, 3), (2, 2), (1, 1)]:
    Xi, _, Yi = core.generate_X(p, q, v, y, k0=3)   # спільний старт -> однакове N
    for name, th in (("МНК", core.mnk(Xi, Yi)), ("РМНК", core.rmnk(Xi, Yi)[0])):
        S, R2, IKA = core.metrics(Yi, core.predict(Xi, th), p, q)
        print("АРКС(%d,%d) %-5s  S = %12.6f   R2 = %.6f   IKA = %10.1f" % (p, q, name, S, R2, IKA))

# ---------------------------------------------------- 7. збереження даних
head("7. Дані для частини B")
os.makedirs(DATA, exist_ok=True)
np.savetxt(os.path.join(DATA, "v.txt"), v, fmt="%.12f")
np.savetxt(os.path.join(DATA, "y.txt"), y, fmt="%.12f")
print("збережено:", os.path.join(DATA, "v.txt"), "та y.txt  (по %d значень, seed = %d)" % (N, SEED))
