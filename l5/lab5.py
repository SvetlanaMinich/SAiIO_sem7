def topological_sort(graph, n):
    """
    Топологическая сортировка вершин графа
    graph: словарь смежности {вершина: [(соседняя_вершина, длина_дуги), ...]}
    n: количество вершин
    Возвращает список вершин в топологическом порядке
    """
    in_degree = {i: 0 for i in range(1, n + 1)}
    
    # Подсчитываем входящие степени
    for v in graph:
        for neighbor, _ in graph[v]:
            in_degree[neighbor] += 1
    
    # Находим вершины без входящих дуг
    queue = [v for v in in_degree if in_degree[v] == 0]
    topo_order = []
    
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        
        if v in graph:
            for neighbor, _ in graph[v]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    return topo_order


def find_longest_path(graph, n, s, t):
    """
    Находит наидлиннейший путь от вершины s до вершины t
    graph: словарь смежности {вершина: [(соседняя_вершина, длина_дуги), ...]}
    n: количество вершин
    s: начальная вершина
    t: конечная вершина
    Возвращает (длина_пути, путь) или (None, None) если путь не существует
    """
    # Топологическая сортировка
    topo_order = topological_sort(graph, n)
    
    # Находим позиции s и t в топологическом порядке
    try:
        k = topo_order.index(s)
        l = topo_order.index(t)
    except ValueError:
        return None, None
    
    # Если s правее t, то путь невозможен
    if k > l:
        return None, None
    
    # Инициализация
    OPT = {}
    x = {}  # Предпоследняя вершина в пути
    
    # Базовый случай
    OPT[s] = 0
    
    # Динамическое программирование
    for i in range(k, l + 1):
        v_i = topo_order[i]
        
        if v_i == s:
            continue
        
        # Находим все вершины, из которых есть дуга в v_i
        incoming = []
        for v in graph:
            for neighbor, length in graph[v]:
                if neighbor == v_i and v in OPT:
                    incoming.append((v, length))
        
        # Если нет входящих дуг или все предыдущие вершины недостижимы
        if not incoming:
            OPT[v_i] = float('-inf')
        else:
            # Находим максимум
            max_length = float('-inf')
            best_vertex = None
            
            for v_prev, edge_length in incoming:
                if OPT[v_prev] != float('-inf'):
                    current_length = OPT[v_prev] + edge_length
                    if current_length > max_length:
                        max_length = current_length
                        best_vertex = v_prev
            
            OPT[v_i] = max_length
            if best_vertex is not None:
                x[v_i] = best_vertex
    
    # Проверяем достижимость
    if t not in OPT or OPT[t] == float('-inf'):
        return None, None
    
    # Восстановление пути (обратный ход)
    path = []
    current = t
    while current != s:
        path.append(current)
        if current not in x:
            break
        current = x[current]
    path.append(s)
    path.reverse()
    
    return OPT[t], path


def print_result(graph, n, s, t):
    """Выводит результат поиска наидлиннейшего пути"""
    print(f"\n{'='*60}")
    print(f"Граф с {n} вершинами:")
    for v in sorted(graph.keys()):
        print(f"  {v} -> {graph[v]}")
    print(f"Начальная вершина: {s}, Конечная вершина: {t}")
    
    length, path = find_longest_path(graph, n, s, t)
    
    if length is None:
        print(f"Путь из вершины {s} в вершину {t} не существует")
    else:
        print(f"Длина наидлиннейшего пути: {length}")
        print(f"Наидлиннейший путь: {' -> '.join(map(str, path))}")
    print(f"{'='*60}")


def main():
    # Пример 1: Простой граф из методички
    # Граф: 1->2(5), 1->3(3), 2->3(2), 2->4(6), 3->4(4), 3->5(2), 4->5(1)
    print("ПРИМЕР 1: Граф из методички")
    graph1 = {
        1: [(2, 5), (3, 3)],
        2: [(3, 2), (4, 6)],
        3: [(4, 4), (5, 2)],
        4: [(5, 1)],
        5: []
    }
    n1 = 5
    s1, t1 = 1, 5
    print_result(graph1, n1, s1, t1)
    
    # Пример 2: Граф с недостижимой вершиной
    print("\nПРИМЕР 2: Граф с недостижимой вершиной")
    graph2 = {
        1: [(2, 3)],
        2: [(3, 4)],
        3: [],
        4: [(5, 2)],
        5: []
    }
    n2 = 5
    s2, t2 = 1, 5
    print_result(graph2, n2, s2, t2)
    
    # Пример 3: Граф где s правее t
    print("\nПРИМЕР 3: Начальная вершина правее конечной")
    graph3 = {
        1: [(2, 5)],
        2: [(3, 3)],
        3: [(4, 2)],
        4: []
    }
    n3 = 4
    s3, t3 = 3, 1
    print_result(graph3, n3, s3, t3)
    
    # Пример 4: Более сложный граф
    print("\nПРИМЕР 4: Сложный граф с множеством путей")
    graph4 = {
        1: [(2, 10), (3, 5)],
        2: [(4, 1), (5, 2)],
        3: [(2, 3), (5, 9)],
        4: [(6, 4)],
        5: [(4, 6), (6, 2)],
        6: []
    }
    n4 = 6
    s4, t4 = 1, 6
    print_result(graph4, n4, s4, t4)


if __name__ == "__main__":
    main()