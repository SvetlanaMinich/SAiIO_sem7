from collections import deque


def build_directed_graph(edges, V1, V2):
    """
    Строит направленный граф G* из неориентированного двудольного графа
    edges: список ребер [(u, v), ...]
    V1, V2: доли графа
    Возвращает граф в виде словаря смежности
    """
    graph = {}
    
    # Инициализация
    graph['s'] = []
    graph['t'] = []
    
    for v in V1:
        graph[v] = []
    for v in V2:
        graph[v] = []
    
    # Шаг 1: Ориентируем ребра слева направо (из V1 в V2)
    for u, v in edges:
        if u in V1 and v in V2:
            graph[u].append(v)
        elif v in V1 and u in V2:
            graph[v].append(u)
    
    # Шаг 2: Добавляем дуги из s в V1 и из V2 в t
    for u in V1:
        graph['s'].append(u)
    
    for v in V2:
        graph[v].append('t')
    
    return graph


def find_path_bfs(graph, start, end):
    """
    Поиск пути от start до end с помощью BFS
    Возвращает путь в виде списка вершин или None
    """
    if start not in graph:
        return None
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            # Восстанавливаем путь
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


def update_graph(graph, path):
    """
    Шаг 5: Корректирует граф G* согласно найденному пути
    - Удаляет первую и последнюю дуги пути
    - Меняет ориентацию промежуточных дуг
    """
    if len(path) < 2:
        return
    
    # Удаляем первую дугу (s -> path[1])
    if path[0] in graph and path[1] in graph[path[0]]:
        graph[path[0]].remove(path[1])
    
    # Удаляем последнюю дугу (path[-2] -> t)
    if path[-2] in graph and path[-1] in graph[path[-2]]:
        graph[path[-2]].remove(path[-1])
    
    # Меняем ориентацию промежуточных дуг
    for i in range(1, len(path) - 2):
        u = path[i]
        v = path[i + 1]
        
        # Удаляем дугу (u, v)
        if u in graph and v in graph[u]:
            graph[u].remove(v)
        
        # Добавляем обратную дугу (v, u)
        if v not in graph:
            graph[v] = []
        if u not in graph[v]:
            graph[v].append(u)


def extract_matching(graph, V1, V2):
    """
    Шаг 4: Извлекает паросочетание из графа G*
    Ищет дуги из V2 в V1
    """
    matching = []
    
    for v in V2:
        if v in graph:
            for u in graph[v]:
                if u in V1:
                    matching.append((u, v))
    
    return matching


def find_maximum_matching(edges, V1, V2):
    """
    Находит наибольшее паросочетание в двудольном графе
    edges: список ребер [(u, v), ...]
    V1, V2: доли графа (списки или множества)
    Возвращает список ребер паросочетания
    """
    V1 = set(V1)
    V2 = set(V2)
    
    # Шаг 1-2: Построение направленного графа G*
    graph = build_directed_graph(edges, V1, V2)
    
    iteration = 0
    print(f"\nНачальный граф G*:")
    print_graph_state(graph, V1, V2)
    
    # Шаги 3-5: Итеративный поиск путей и корректировка графа
    while True:
        iteration += 1
        
        # Шаг 3: Поиск (s, t)-пути
        path = find_path_bfs(graph, 's', 't')
        
        if path is None:
            # Шаг 4: Путь не найден - извлекаем паросочетание
            print(f"\nИтерация {iteration}: (s, t)-путь не найден")
            print("Алгоритм завершает работу")
            matching = extract_matching(graph, V1, V2)
            return matching
        
        print(f"\nИтерация {iteration}: Найден (s, t)-путь: {' -> '.join(map(str, path))}")
        
        # Шаг 5: Корректировка графа
        update_graph(graph, path)
        print(f"Граф после корректировки:")
        print_graph_state(graph, V1, V2)


def print_graph_state(graph, V1, V2):
    """Выводит текущее состояние графа"""
    print("  Дуги из s:", graph.get('s', []))
    
    for v in sorted(V1):
        if v in graph and graph[v]:
            print(f"  Дуги из {v}:", graph[v])
    
    for v in sorted(V2):
        if v in graph and graph[v]:
            print(f"  Дуги из {v}:", graph[v])


def visualize_matching(edges, V1, V2, matching):
    """Визуализирует результат"""
    print(f"\n{'='*60}")
    print("ИСХОДНЫЙ ГРАФ:")
    print(f"Доля V1: {sorted(V1)}")
    print(f"Доля V2: {sorted(V2)}")
    print(f"Рёбра: {edges}")
    
    print(f"\nНАИБОЛЬШЕЕ ПАРОСОЧЕТАНИЕ:")
    print(f"Количество рёбер: {len(matching)}")
    print(f"Рёбра паросочетания: {sorted(matching)}")
    
    # Проверка корректности
    vertices_in_matching = set()
    for u, v in matching:
        if u in vertices_in_matching or v in vertices_in_matching:
            print("ОШИБКА: Найдены рёбра с общей вершиной!")
        vertices_in_matching.add(u)
        vertices_in_matching.add(v)
    
    print(f"{'='*60}")


def main():
    # Пример 1: Граф из методички
    print("="*60)
    print("ПРИМЕР 1: Граф из методички")
    print("="*60)
    
    V1_1 = ['a', 'b', 'c']
    V2_1 = ['x', 'y', 'z']
    edges_1 = [
        ('a', 'x'),
        ('a', 'y'),
        ('b', 'x'),
        ('c', 'y'),
        ('c', 'z')
    ]
    
    matching_1 = find_maximum_matching(edges_1, V1_1, V2_1)
    visualize_matching(edges_1, V1_1, V2_1, matching_1)
    
    # Пример 2: Полное паросочетание
    print("\n" + "="*60)
    print("ПРИМЕР 2: Полное паросочетание")
    print("="*60)
    
    V1_2 = [1, 2, 3]
    V2_2 = [4, 5, 6]
    edges_2 = [
        (1, 4),
        (1, 5),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 6)
    ]
    
    matching_2 = find_maximum_matching(edges_2, V1_2, V2_2)
    visualize_matching(edges_2, V1_2, V2_2, matching_2)
    
    # Пример 3: Неполное паросочетание
    print("\n" + "="*60)
    print("ПРИМЕР 3: Неполное паросочетание")
    print("="*60)
    
    V1_3 = ['A', 'B', 'C', 'D']
    V2_3 = ['X', 'Y', 'Z']
    edges_3 = [
        ('A', 'X'),
        ('B', 'X'),
        ('B', 'Y'),
        ('C', 'Y'),
        ('D', 'Z')
    ]
    
    matching_3 = find_maximum_matching(edges_3, V1_3, V2_3)
    visualize_matching(edges_3, V1_3, V2_3, matching_3)
    
    # Пример 4: Граф с изолированными вершинами
    print("\n" + "="*60)
    print("ПРИМЕР 4: Граф с изолированными вершинами")
    print("="*60)
    
    V1_4 = [1, 2, 3, 4]
    V2_4 = [5, 6, 7, 8]
    edges_4 = [
        (1, 5),
        (2, 6),
        (2, 7)
    ]
    
    matching_4 = find_maximum_matching(edges_4, V1_4, V2_4)
    visualize_matching(edges_4, V1_4, V2_4, matching_4)


if __name__ == "__main__":
    main()