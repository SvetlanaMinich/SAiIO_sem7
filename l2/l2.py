import numpy as np

def is_invertible_and_inverse(A_inv, x, i):
    n = A_inv.shape[0]
    l = A_inv @ x
    if abs(l[i]) < 1e-12:
        return False, None
    la = l.copy()
    la[i] = -1
    lb = (-1.0 / l[i]) * la
    lb = lb.flatten()
    Q = np.eye(n)
    Q[:, i] = lb
    B_inv = Q @ A_inv
    return True, B_inv

def simplex_method(c, A, b, B, tol=1e-9):
    """
    Основная фаза симплекса.
    Возвращает: (x, B, A_inv) при успехе,
                или строку с сообщением при неограниченности/ошибке.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=float)
    AB = A[:, B]
    try:
        A_inv = np.linalg.inv(AB)
    except np.linalg.LinAlgError:
        return "Начальная базисная матрица необратима.", None, None

    x[B] = A_inv @ b  # базисные значения

    while True:
        AB = A[:, B]
        cB = c[B]
        u = cB @ A_inv                  # потенциалы (строка)
        delta = u @ A - c               # оценки (строка)

        # оптимальность по максимуму: все оценки >= 0
        if np.all(delta >= -tol):
            # Отдаём также A_inv чтобы можно было строить срез
            return x, B, A_inv

        # выбрать небазис с отрицательной оценкой (улучшающая переменная)
        j0_candidates = np.where(delta < -tol)[0]
        j0 = j0_candidates[0]

        z = A_inv @ A[:, j0]  # направление (вектор длины m)

        theta = np.full(m, np.inf, dtype=float)
        positive = z > tol
        if np.any(positive):
            theta[positive] = x[B][positive] / z[positive]

        if np.all(theta == np.inf):
            return "Целевой функционал задачи не ограничен сверху на множестве допустимых планов.", None, None

        k = np.argmin(theta)  # строка, которая уйдёт из базиса
        j_star = B[k]
        B = B.copy()
        B[k] = j0

        success, A_inv = is_invertible_and_inverse(A_inv, A[:, j0], k)
        if not success:
            return "Новая базисная матрица необратима.", None, None

        # обновление x: по правилу смены базиса
        x[j0] = theta[k]
        for i in range(m):
            if i != k:
                x[B[i]] -= theta[k] * z[i]
        x[j_star] = 0.0

def fractional_part(x, tol=1e-12):
    # корректно работает с отрицательными числами
    frac = x - np.floor(x)
    # если значение почти целое — считаем дробной частью 0
    frac[np.abs(frac) < tol] = 0.0
    return frac

def gomory_cut_from_basis(A, B, A_inv, x, tol=1e-9):
    """
    Построить один срез Гомори по первому дробному базисному элементу.
    Возвращает словарь с полями:
      - coeffs : вектор длины n (коэффициенты при исходных x)
      - rhs    : правый член (число)  (равно fract(b_k))
      - new_var: имя новой переменной (строка), например 's'
      - as_ineq: True (то есть sum coeffs * x >= rhs)
    """
    m, n = A.shape
    # вычислим базисную правую часть b_B = A_inv @ b, но у нас x[B] уже равен этому
    b_B = x[B]

    # Найти первую базисную с дробной частью
    fracs = fractional_part(b_B, tol=tol)
    idx = np.where(fracs > tol)[0]
    if idx.size == 0:
        return {"status": "integer_solution", "message": "Нет дробных базисных. Решение целочисленное."}

    k = int(idx[0])   # позиция в базисе (0..m-1)
    b_frac = fracs[k]

    # построим A_N — матрицу небазисных столбцов и список индексов N
    all_idx = list(range(n))
    B_list = list(B)
    N_list = [j for j in all_idx if j not in B_list]

    if len(N_list) == 0:
        # нет небазисных — срез не строится в классическом виде
        return {"status": "no_nonbasic", "message": "Нет небазисных переменных для формирования среза."}

    # Q = A_B^{-1} * A_N — m x |N|
    A_N = A[:, N_list]
    Q = A_inv @ A_N   # m x |N|
    ell = Q[k, :]     # k-я строка (коэффициенты при небазисных в выражении x_{j_k})

    # дробные части коэффициентов
    ell_frac = fractional_part(ell, tol=tol)

    # сформируем вектор коэффициентов длины n (для исходных x1..xn)
    coeffs = np.zeros(n, dtype=float)
    for p, j in enumerate(N_list):
        coeffs[j] = ell_frac[p]
    # для базисных переменных коэффициции 0 (в классическом срезе они не участвуют)

    # Возвращаем срез в удобной форме
    return {
        "status": "cut",
        "coeffs": coeffs,
        "rhs": float(b_frac),
        "new_var": "s",          # новая переменная s >= 0 (для представления как равенство)
        "as_ineq": True,         # означает sum coeffs * x >= rhs
        "k_basis_pos": k,
        "B_index": int(B[k])
    }

def print_gomory_cut_book(cut):
    """
    Форматируем и печатаем срез в виде (коэффициенты при x) - s = rhs
    cut: словарь как возвращает gomory_cut_from_basis
    """
    if cut.get("status") != "cut":
        print("Нет среза для печати:", cut.get("message", cut))
        return

    coeffs = cut["coeffs"]
    rhs = cut["rhs"]
    n = len(coeffs)

    terms = []
    for j in range(n):
        a = coeffs[j]
        if abs(a) < 1e-12:
            continue
        var_name = f"x{j+1}"   # человекочитаемая нумерация с 1
        # если коэффициент равен 1, пишем просто xj; если -1, пишем -xj
        # но у нас коэффициенты положительные по построению среза, оставим общий формат
        terms.append(f"{a}*{var_name}")

    if len(terms) == 0:
        left = "0"
    else:
        left = " + ".join(terms)

    print(f"{left} - s = {rhs}   (s >= 0)")


# --- пример использования ---
if __name__ == "__main__":
    # пример из условия лабы
    c = np.array([0.0, 1.0, 0.0, 0.0])
    A = np.array([
        [ 3.0,  2.0, 1.0, 0.0],
        [-3.0,  2.0, 0.0, 1.0]
    ])
    b = np.array([6.0, 0.0])
    # В стандартной постановке (когда все ограничения уже приведены к виду «=» и добавлены дополнительные переменные) обычно базисными стартуют именно добавленные переменные потому что они образуют «единичную матрицу».
    B = np.array([2, 3])  # x3 и x4

    res = simplex_method(c, A, b, B)
    if isinstance(res, tuple):
        x_opt, B_opt, A_inv = res
        print("LP optimal x:", x_opt)
        cut = gomory_cut_from_basis(A, B_opt, A_inv, x_opt)
        print(f"Coefficients of Gomory cut: {cut['coeffs']}")
        print_gomory_cut_book(cut)
    else:
        print("Simplex returned:", res)
