# ЛР №1 (АЧР), бригада 11. Частина A — обчислювальне ядро.
# Модель: y(k) = 0.4 + 0.05*y(k-1) - 0.05*y(k-2) + 0.5*y(k-3)
#                    + v(k) + 0.4*v(k-1) + 0.5*v(k-2) + 0.1*v(k-3)
# theta = [a0, a1..ap, c0, b1..bq],  x(k) = [1, y(k-1)..y(k-p), v(k)..v(k-q)]
# c0 — коефіцієнт при v(k), істинне значення 1. Для IKA: n = p + q + 1.

import numpy as np

# ---------------------------------------------------------------- варіант 11
A_TRUE = np.array([0.4, 0.05, -0.05, 0.5])   # a0, a1, a2, a3
B_TRUE = np.array([0.4, 0.5, 0.1])           # b1, b2, b3
P_TRUE, Q_TRUE = 3, 3


def theta_true(p=P_TRUE, q=Q_TRUE):
    # істинні параметри в порядку стовпців X
    return np.concatenate([A_TRUE[:p + 1], [1.0], B_TRUE[:q]])


def check_stability(a=A_TRUE):
    # стійкість: sum|a_i| < 1, a0 не враховується
    s_abs = float(np.abs(a[1:]).sum())
    s_raw = float(a[1:].sum())
    return s_abs, s_raw, s_abs < 1.0


# ------------------------------------------------------- генерування процесу
def generate_noise(N, seed=11, sigma=1.0):
    # білий шум v(k) ~ N(0, sigma^2)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, N)


def generate_noise_uniform12(N, seed=11):
    # той самий шум через n = sum(xi_1..xi_12) - 6
    # xi ~ U(0,1), а не U(-1,1) як надруковано у вказівках: інакше середнє -6
    rng = np.random.default_rng(seed)
    xi = rng.uniform(0.0, 1.0, (N, 12))
    return xi.sum(axis=1) - 6.0


def generate_series(v, a=A_TRUE, b=B_TRUE):
    # еталонний ряд y(k) рекурентно; перші max(p,q) значень = 0
    p = len(a) - 1
    q = len(b)
    m = max(p, q)
    N = len(v)
    y = np.zeros(N)
    for k in range(m, N):
        acc = a[0]
        for i in range(1, p + 1):
            acc += a[i] * y[k - i]
        acc += v[k]
        for j in range(1, q + 1):
            acc += b[j - 1] * v[k - j]
        y[k] = acc
    return y


# ------------------------------------------------- матриця вимірів та методи
def generate_X(p, q, v, y, k0=None):
    # рядок X: [1, y(k-1)..y(k-p), v(k), v(k-1)..v(k-q)]
    # k0 — спільний старт для всіх моделей (частина B), щоб N збігалося
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(v) != len(y):
        raise ValueError("ряди v та y мають бути однакової довжини (див. align_loaded)")

    finite = np.flatnonzero(np.isfinite(y))
    first = int(finite[0]) if finite.size else 0
    margin = max(first + p, q)          # NaN на початку y -> рівняння відкидаються
    if k0 is not None:
        margin = max(margin, int(k0))

    ks = np.arange(margin, len(v))
    rows = [np.ones(len(ks))]
    rows += [y[ks - i] for i in range(1, p + 1)]
    rows += [v[ks - j] for j in range(0, q + 1)]
    X = np.column_stack(rows)
    Y = y[ks]
    return X, margin, Y


def mnk(X, Y):
    # theta = (X^T X)^-1 X^T Y; NaN якщо матриця вироджена
    try:
        theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    except np.linalg.LinAlgError:
        theta = np.full(X.shape[1], np.nan)
    return theta


def rmnk(X, Y, beta=10.0):
    # рекурсивний МНК без обернення матриці, формули (2.16)-(2.18)
    n_rows, n_params = X.shape
    theta = np.zeros(n_params)              # theta(0) = 0
    thetas = np.zeros((n_rows, n_params))   # історія -> перехідний процес
    P = np.identity(n_params) * beta        # P(0) = beta * I

    for i in range(n_rows):
        x_i = X[i, :].reshape(1, n_params)
        numerator = P @ x_i.T @ x_i @ P
        denominator = 1.0 + x_i @ P @ x_i.T
        P = P - numerator / denominator
        dif = Y[i] - x_i @ theta            # похибка прогнозу
        theta = theta + (P @ x_i.T * dif).ravel()
        thetas[i, :] = theta
    return theta, thetas


def mnk_transient(X, Y):
    # МНК на вибірці, що зростає: theta по перших k рядках (п.3 протоколу)
    n_rows, n_params = X.shape
    thetas = np.full((n_rows, n_params), np.nan)
    for k in range(n_params, n_rows + 1):
        thetas[k - 1, :] = mnk(X[:k], Y[:k])
    return thetas


# ------------------------------------------------------------------ критерії
def predict(X, theta):
    return X @ theta


def metrics(Y, Y_hat, p, q):
    # S = sum e^2,  R^2 = var(y_hat)/var(y),  IKA = N*ln(S/N) + 2n
    Y = np.asarray(Y, dtype=float)
    Y_hat = np.asarray(Y_hat, dtype=float)
    if not np.all(np.isfinite(Y_hat)):
        return np.nan, np.nan, np.nan

    e = Y - Y_hat
    N = len(Y)
    S = float(e @ e)
    R2 = float(np.var(Y_hat) / np.var(Y))
    n = p + q + 1
    IKA = float(N * np.log(max(S, 1e-300) / N) + 2 * n)
    return S, R2, IKA


# --------------------------------------------------------- робота з файлами
def load_series(path):
    return np.loadtxt(path)


def align_loaded(v, y):
    # у даних викладача y.txt починається з k = 3 -> доповнюємо початок NaN
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = len(v) - len(y)
    if offset < 0:
        raise ValueError("ряд y довший за v")
    return v, np.concatenate([np.full(offset, np.nan), y])
