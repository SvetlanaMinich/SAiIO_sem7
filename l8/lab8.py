from collections import deque


def labeling_method(G_f, s, t):
    """
    Метод пометок для поиска (s, t)-пути во вспомогательной сети
    Возвращает (путь, множество_достижимых_вершин)
    """
    # Шаг 1: Инициализация
    Q = deque()
    X = set()
    labels = {}
    
    # Шаг 2: Помечаем исток
    Q.append(s)
    X.add(s)
    labels[s] = None
    
    print(f"\nМетод пометок:")
    print(f"  Начало: помечаем вершину {s}")
    
    # Шаг 3-5: Обход в ширину
    while Q:
        v = Q.popleft()
        print(f"  Обрабатываем вершину {v}")
        
        # Шаг 5: Рассматриваем дуги из v
        if v in G_f:
            for u, capacity in G_f[v].items():
                # Рассматриваем только дуги с ненулевой пропускной способностью
                if capacity > 0 and u not in labels:
                    labels[u] = (v, u)
                    X.add(u)
                    Q.append(u)
                    print(f"    Помечаем вершину {u}, метка: ({v}, {u})")
                    
                    # Если достигли стока, можно остановиться
                    if u == t:
                        print(f"  Сток {t} достигнут!")
                        path = reconstruct_path(labels, s, t)
                        return path, X
    
    print(f"  Сток {t} не достижим из {s}")
    return None, X


def reconstruct_path(labels, s, t):
    """Восстанавливает путь от s до t по меткам"""
    if t not in labels:
        return None
    
    path = []
    current = t
    
    while current != s:
        edge = labels[current]
        if edge is None:
            break
        path.append(edge)
        current = edge[0]
    
    path.reverse()
    return path


def build_residual_network(capacity, flow):
    """
    Строит вспомогательную сеть G_f
    Возвращает граф с пропускными способностями c_f(a) = c(a) - f(a) + f(ā)
    """
    G_f = {}
    
    # Проходим по всем дугам
    for u in capacity:
        if u not in G_f:
            G_f[u] = {}
        
        for v, cap in capacity[u].items():
            # Прямая дуга: c_f(u,v) = c(u,v) - f(u,v) + f(v,u)
            f_uv = flow.get(u, {}).get(v, 0)
            f_vu = flow.get(v, {}).get(u, 0)
            c_f = cap - f_uv + f_vu
            
            if c_f != 0:  # Включаем все дуги, в том числе с нулевой пропускной способностью
                G_f[u][v] = c_f
    
    return G_f


def ford_fulkerson(capacity, s, t):
    """
    Алгоритм Форда-Фалкерсона для поиска максимального потока
    capacity: словарь словарей {u: {v: c(u,v), ...}, ...}
    s: исток
    t: сток
    Возвращает: (максимальный_поток, мощность_потока, минимальный_разрез)
    """
    # Шаг 1: Инициализация нулевого потока
    flow = {}
    for u in capacity:
        flow[u] = {}
        for v in capacity[u]:
            flow[u][v] = 0
    
    iteration = 0
    total_flow = 0
    
    print("="*70)
    print("АЛГОРИТМ ФОРДА-ФАЛКЕРСОНА")
    print("="*70)
    print(f"\nШаг 1: Инициализация нулевого потока")
    
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ИТЕРАЦИЯ {iteration}")
        print(f"{'='*70}")
        
        # Шаг 2: Построение вспомогательной сети G_f
        G_f = build_residual_network(capacity, flow)
        
        print(f"\nТекущий поток:")
        print_flow(flow, capacity)
        
        print(f"\nВспомогательная сеть G_f (пропускные способности c_f):")
        print_network(G_f)
        
        # Шаги 3-4: Поиск (s, t)-пути методом пометок
        path, reachable = labeling_method(G_f, s, t)
        
        # Шаг 3: Проверка существования пути
        if path is None:
            print(f"\n(s, t)-путь не найден. Текущий поток максимальный!")
            print(f"\nМножество достижимых из {s} вершин: {sorted(reachable)}")
            
            # Находим минимальный разрез
            unreachable = set(capacity.keys()) - reachable
            if t in capacity:
                unreachable.add(t)
            
            min_cut = (sorted(reachable), sorted(unreachable))
            print(f"Минимальный разрез: ({min_cut[0]}, {min_cut[1]})")
            
            return flow, total_flow, min_cut
        
        print(f"\nНайден путь: {format_path(path)}")
        
        # Шаг 5: Находим минимальную пропускную способность на пути
        theta = min(G_f[u][v] for u, v in path)
        print(f"Минимальная пропускная способность θ = {theta}")
        
        # Шаг 6-7: Увеличиваем поток
        for u, v in path:
            # f'(u,v) = max(0, f(u,v) - f(v,u) + f_P(u,v) - f_P(v,u))
            # Для прямых дуг пути: f_P(u,v) = θ, f_P(v,u) = 0
            # f'(u,v) = max(0, f(u,v) - f(v,u) + θ)
            
            f_uv = flow.get(u, {}).get(v, 0)
            f_vu = flow.get(v, {}).get(u, 0)
            
            new_f_uv = max(0, f_uv - f_vu + theta)
            new_f_vu = max(0, f_vu - f_uv - theta)
            
            if u not in flow:
                flow[u] = {}
            if v not in flow:
                flow[v] = {}
            
            flow[u][v] = new_f_uv
            flow[v][u] = new_f_vu
        
        total_flow += theta
        print(f"\nПоток увеличен на {theta}, общая мощность потока: {total_flow}")


def format_path(path):
    """Форматирует путь для вывода"""
    if not path:
        return ""
    result = str(path[0][0])
    for u, v in path:
        result += f" -> {v}"
    return result


def print_network(graph):
    """Выводит граф (сеть или вспомогательную сеть)"""
    for u in sorted(graph.keys()):
        if graph[u]:
            edges = [f"{v}(c={c})" for v, c in sorted(graph[u].items()) if c > 0]
            if edges:
                print(f"  {u}: {', '.join(edges)}")


def print_flow(flow, capacity):
    """Выводит текущий поток в формате f/c"""
    for u in sorted(capacity.keys()):
        edges = []
        for v in sorted(capacity[u].keys()):
            f = flow.get(u, {}).get(v, 0)
            c = capacity[u][v]
            if c > 0 or f > 0:
                edges.append(f"{v}: {f}/{c}")
        if edges:
            print(f"  {u}: {', '.join(edges)}")


def calculate_flow_value(flow, capacity, s):
    """Вычисляет мощность потока (поток из истока)"""
    total = 0
    if s in flow:
        for v in flow[s]:
            if v in capacity.get(s, {}):
                total += flow[s][v]
    return total


def main():
    # Пример: Сеть из условия
    print("="*70)
    print("ПРИМЕР")
    print("="*70)
    
    capacity1 = {
        's': {'a': 3, 'b': 2},
        'a': {'t': 1, 'b': 2},
        'b': {'t': 2},
        't': {}
    }
    
    print("\nИсходная сеть (пропускные способности):")
    print_network(capacity1)
    
    flow1, max_flow1, min_cut1 = ford_fulkerson(capacity1, 's', 't')
    
    print(f"\n{'='*70}")
    print(f"РЕЗУЛЬТАТ")
    print(f"{'='*70}")
    print(f"Максимальный поток: {max_flow1}")
    print(f"\nРаспределение потока:")
    print_flow(flow1, capacity1)


if __name__ == "__main__":
    main()