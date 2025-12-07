import numpy as np

def is_invertible_and_inverse(A_inv, x, i):
    """
    Пересчёт обратной матрицы A_B^{-1}, если в базисе 
    меняется один столбец.

    :param A_inv: старая обратная матрица A_B^{-1}
    :param x: новый столбец матрицы A, который входит в базис
    :param i: номер строки, где происходит замена базисного столбца
    """
    n = A_inv.shape[0]
    # l = новый столбец
    l = A_inv @ x
    if abs(l[i]) < 1e-12:  # если на позиции i стоит 0 → обратимость нарушена
        return False, None
    la = l.copy()
    la[i] = -1
    # формула обновления
    lb = (-1.0 / l[i]) * la
    lb = lb.flatten()
    Q = np.eye(n)
    Q[:, i] = lb
    # новое A_B^{-1}
    B_inv = Q @ A_inv
    return True, B_inv

def simplex_method(c, A, b, B, tol=1e-9):
    """
    Реализация основной фазы симплекс-метода.

    :param c: вектор коэффициентов целевой функции
    :param A: матрица ограничений
    :param b: вектор правых частей
    :param B: список индексов базисных переменных (нумерация с 0)
    :return: (оптимальный план x, список базисных переменных B, A_B^{-1})
    """
    m, n = A.shape
    # решение x = 0
    x = np.zeros(n, dtype=float)

    # выделяем матрицу базисных столбцов
    AB = A[:, B]
    try:
        A_inv = np.linalg.inv(AB)
    except np.linalg.LinAlgError:
        return "Initial basis matrix is not invertible.", None, None

    # вычисляем значения базисных переменных 
    x[B] = A_inv @ b  

    while True:
        # Вычисляем вектора движения
        cB = c[B]
        u = cB @ A_inv          # вектор потенциалов
        delta = u @ A - c       # оценки (на сколько двигаемся)

        # проверяем оптимальность
        if np.all(delta >= -tol):
            return x, B, A_inv

        # выбираем первую небазисную переменную с отрицательной оценкой
        j0_candidates = np.where(delta < -tol)[0]
        j0 = j0_candidates[0]

        # направление изменения
        z = A_inv @ A[:, j0]

        # считаем минимальное отношение theta
        theta = np.full(m, np.inf, dtype=float)
        positive = z > tol
        if np.any(positive):
            theta[positive] = x[B][positive] / z[positive]

        # если все z <= 0, то задача неограничена
        if np.all(theta == np.inf):
            return "Objective function is unbounded above.", None, None

        # обновляем базис
        k = np.argmin(theta)  
        j_star = B[k]
        B = B.copy()
        B[k] = j0 

        # обновляем обратную матрицу
        success, A_inv = is_invertible_and_inverse(A_inv, A[:, j0], k)
        if not success:
            return "New basis matrix is not invertible.", None, None

        # обновляем решение x
        x[j0] = theta[k]
        for i in range(m):
            if i != k:
                x[B[i]] -= theta[k] * z[i]
        x[j_star] = 0.0


def fractional_part(x, tol=1e-12):
    """
    Берём дробную часть числа: {t} = t - floor(t).
    """
    frac = x - np.floor(x)
    frac[np.abs(frac) < tol] = 0.0
    return frac

def gomory_cut_from_basis(A, B, A_inv, x, tol=1e-9):
    """
    Строим один срез Гомори:
    - выбираем первую базисную переменную с дробной частью,
    - формируем неравенство sum {ell_j} * x_j >= {b_k}.

    :return: коэффициенты среза
    """
    m, n = A.shape
    # базисные значения
    b_B = x[B]

    # ищем первую дробную базисную
    fracs = fractional_part(b_B, tol=tol)
    idx = np.where(fracs > tol)[0]
    if idx.size == 0:
        return {"status": "int solution", "message": "All basis variables are integer."}

    k = int(idx[0])   # позиция дробной базисной переменной
    b_frac = fracs[k] # её дробная часть

    # список небазисных
    all_idx = list(range(n))
    N_list = [j for j in all_idx if j not in B]

    if len(N_list) == 0:
        return {"status": "no_nonbasic", "message": "No nonbasic variables."}

    # считаем коэффициенты при небазисных переменных
    A_N = A[:, N_list]
    Q = A_inv @ A_N
    ell = Q[k, :]

    # берём дробные части коэффициентов
    ell_frac = fractional_part(ell, tol=tol)

    # формируем вектор коэффициентов при x
    coeffs = np.zeros(n, dtype=float)
    for p, j in enumerate(N_list):
        coeffs[j] = ell_frac[p]

    return {
        "status": "cut",
        "coeffs": coeffs,
        "rhs": float(b_frac)
    }

def print_gomory_cut_book(cut):
    """
    Печать среза в формате из методички:
    (сумма дробных коэффициентов при x) - s = {b_k},  s >= 0
    """
    if cut.get("status") != "cut":
        print("No cut:", cut.get("message", cut))
        return

    coeffs = cut["coeffs"]
    rhs = cut["rhs"]
    n = len(coeffs)

    terms = []
    for j in range(n):
        a = coeffs[j]
        if abs(a) < 1e-12:
            continue
        var_name = f"x{j+1}"  
        terms.append(f"{a}*{var_name}")

    left = " + ".join(terms) if terms else "0"
    print(f"{left} - s = {rhs}   (s >= 0)")


if __name__ == "__main__":
    c = np.array([0.0, 1.0, 0.0, 0.0])
    A = np.array([
        [ 3.0,  2.0, 1.0, 0.0],
        [-3.0,  2.0, 0.0, 1.0]
    ])
    b = np.array([6.0, 0.0])

    # стартовый базис
    B = np.array([2, 3])

    res = simplex_method(c, A, b, B)
    if isinstance(res, tuple):
        x_opt, B_opt, A_inv = res
        print("Optimal LP solution:", x_opt)
        cut = gomory_cut_from_basis(A, B_opt, A_inv, x_opt)
        print("Gomory cut (coeffs):", cut['coeffs'])
        print("Gomory cut:")
        print_gomory_cut_book(cut)
    else:
        print("Simplex returned:", res)