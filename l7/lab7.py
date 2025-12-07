from collections import deque
from fractions import Fraction


def find_path_bfs(graph, start, end):
    """Поиск пути от start до end с помощью BFS"""
    if start not in graph:
        return None
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    
    return None


def build_directed_graph_for_matching(edges, n):
    """Строит направленный граф G* для поиска паросочетания"""
    graph = {i: [] for i in range(1, n + 1)}
    graph['s'] = []
    graph['t'] = []
    
    for i in range(1, n + 1):
        graph['s'].append(i)
        graph[i + n] = []
        graph[i + n].append('t')
    
    # edges содержит пары (i, j) с индексацией с 0, преобразуем в индексацию с 1
    for i, j in edges:
        graph[i + 1].append(j + 1 + n)
    
    return graph


def update_graph_for_matching(graph, path):
    """Корректирует граф по найденному пути"""
    if len(path) < 2:
        return
    
    if path[0] in graph and path[1] in graph[path[0]]:
        graph[path[0]].remove(path[1])
    
    if path[-2] in graph and path[-1] in graph[path[-2]]:
        graph[path[-2]].remove(path[-1])
    
    for i in range(1, len(path) - 2):
        u = path[i]
        v = path[i + 1]
        
        if u in graph and v in graph[u]:
            graph[u].remove(v)
        
        if v not in graph:
            graph[v] = []
        if u not in graph[v]:
            graph[v].append(u)


def find_maximum_matching_internal(edges, n):
    """Находит наибольшее паросочетание и возвращает граф G*"""
    graph = build_directed_graph_for_matching(edges, n)
    
    while True:
        path = find_path_bfs(graph, 's', 't')
        
        if path is None:
            matching = []
            for i in range(n + 1, 2 * n + 1):
                if i in graph:
                    for j in graph[i]:
                        if j != 't' and 1 <= j <= n:
                            matching.append((j, i - n))
            return matching, graph
        
        update_graph_for_matching(graph, path)


def find_reachable_from_s(graph, n):
    """Находит все вершины, достижимые из s в графе G*"""
    if 's' not in graph:
        return set(), set()
    
    queue = deque(['s'])
    visited = {'s'}
    
    while queue:
        current = queue.popleft()
        
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    I_star = set()
    J_star = set()
    
    for v in visited:
        if isinstance(v, int):
            if 1 <= v <= n:
                I_star.add(v)
            elif n + 1 <= v <= 2 * n:
                J_star.add(v - n)
    
    return I_star, J_star


def hungarian_algorithm(C):
    """
    Венгерский алгоритм для решения задачи о назначениях
    C: квадратная матрица стоимостей (list of lists)
    Возвращает список пар (i, j) - индексы выбранных элементов (нумерация с 0)
    """
    n = len(C)
    
    # Используем Fraction для точных вычислений
    alpha = [Fraction(0) for _ in range(n)]
    beta = [Fraction(0) for _ in range(n)]
    
    # Шаг 1: Инициализация допустимого плана
    for j in range(n):
        beta[j] = Fraction(min(C[i][j] for i in range(n)))
    
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ИТЕРАЦИЯ {iteration}")
        print(f"{'='*70}")
        
        print("\nТекущий допустимый план:")
        print("α =", [float(a) if a.denominator != 1 else int(a) for a in alpha])
        print("β =", [float(b) if b.denominator != 1 else int(b) for b in beta])
        
        # Шаг 2: Находим множества J= и J<
        J_eq = []
        for i in range(n):
            for j in range(n):
                if alpha[i] + beta[j] == Fraction(C[i][j]):
                    J_eq.append((i, j))
        
        print(f"\nМножество J= (пары с α_i + β_j = c_ij):")
        print(f"J= = {[(i+1, j+1) for i, j in J_eq]}")
        
        # Шаг 3-4: Построение графа и поиск паросочетания
        matching, graph_star = find_maximum_matching_internal(J_eq, n)
        
        print(f"\nНайдено паросочетание размера {len(matching)}")
        print(f"M = {[(i, j) for i, j in matching]}")
        
        # Шаг 5: Проверка оптимальности
        if len(matching) == n:
            print(f"\n|M| = n = {n}, паросочетание совершенное!")
            print("Алгоритм завершает работу")
            
            # Преобразуем в индексы с 0
            result = [(i - 1, j - 1) for i, j in matching]
            return result
        
        # Шаг 6: Находим достижимые вершины из s
        I_star, J_star = find_reachable_from_s(graph_star, n)
        
        print(f"\nВершины, достижимые из s:")
        print(f"I* = {sorted(I_star)}")
        print(f"J* = {sorted(J_star)}")
        
        # Шаг 7-8: Формируем допустимый план двойственной задачи
        alpha_tilde = []
        beta_tilde = []
        
        for i in range(1, n + 1):
            if i in I_star:
                alpha_tilde.append(Fraction(1))
            else:
                alpha_tilde.append(Fraction(-1))
        
        for j in range(1, n + 1):
            if j in J_star:
                beta_tilde.append(Fraction(-1))
            else:
                beta_tilde.append(Fraction(1))
        
        # Шаг 9: Находим θ
        theta = None
        for i in range(n):
            if (i + 1) in I_star:
                for j in range(n):
                    if (j + 1) not in J_star:
                        value = Fraction(C[i][j] - alpha[i] - beta[j]) / Fraction(2)
                        if theta is None or value < theta:
                            theta = value
        
        print(f"\nθ = {float(theta) if theta.denominator != 1 else int(theta)}")
        
        # Шаг 10: Обновляем допустимый план
        for i in range(n):
            alpha[i] = alpha[i] + theta * alpha_tilde[i]
        
        for j in range(n):
            beta[j] = beta[j] + theta * beta_tilde[j]


def print_matrix_with_solution(C, solution):
    """Выводит матрицу с выделенным решением"""
    n = len(C)
    
    print("\nМатрица с оптимальным решением (выделено *):")
    
    # Заголовок
    print("    ", end="")
    for j in range(n):
        print(f"  [{j+1}]", end="")
    print()
    
    total_cost = 0
    
    for i in range(n):
        print(f"[{i+1}] ", end="")
        for j in range(n):
            if (i, j) in solution:
                print(f"  {C[i][j]:2d}*", end="")
                total_cost += C[i][j]
            else:
                print(f"  {C[i][j]:2d} ", end="")
        print()
    
    print(f"\nОптимальное решение:")
    for i, j in sorted(solution):
        print(f"  Строка {i+1} -> Столбец {j+1}, стоимость = {C[i][j]}")
    
    print(f"\nОбщая стоимость = {total_cost}")


def main():
    # Пример 1: Матрица из методички
    print("="*70)
    print("ПРИМЕР 1: Матрица из методички")
    print("="*70)
    
    C1 = [
        [7, 2, 1, 9, 4],
        [9, 6, 9, 5, 5],
        [3, 8, 3, 1, 8],
        [7, 9, 4, 2, 2],
        [8, 4, 7, 4, 8]
    ]
    
    print("\nИсходная матрица:")
    for row in C1:
        print(row)
    
    solution1 = hungarian_algorithm(C1)
    print_matrix_with_solution(C1, solution1)
    
    # Пример 2: Простая матрица 3x3
    print("\n\n" + "="*70)
    print("ПРИМЕР 2: Простая матрица 3x3")
    print("="*70)
    
    C2 = [
        [4, 2, 8],
        [4, 3, 7],
        [3, 1, 6]
    ]
    
    print("\nИсходная матрица:")
    for row in C2:
        print(row)
    
    solution2 = hungarian_algorithm(C2)
    print_matrix_with_solution(C2, solution2)
    
    # Пример 3: Матрица 4x4
    print("\n\n" + "="*70)
    print("ПРИМЕР 3: Матрица 4x4")
    print("="*70)
    
    C3 = [
        [10, 19, 8, 15],
        [10, 18, 7, 17],
        [13, 16, 9, 14],
        [12, 19, 8, 18]
    ]
    
    print("\nИсходная матрица:")
    for row in C3:
        print(row)
    
    solution3 = hungarian_algorithm(C3)
    print_matrix_with_solution(C3, solution3)


if __name__ == "__main__":
    main()