def knapsack_solver(volumes, values, capacity):
    """
    Решает задачу о целочисленном рюкзаке методом динамического программирования.
    
    Аргументы:
        volumes: список объёмов предметов (v_i)
        values: список ценностей предметов (c_i)
        capacity: вместимость рюкзака (B)
    
    Возвращает:
        tuple: (максимальная ценность, список выбранных предметов (индексы))
    """
    n = len(volumes)
    
    # Инициализация таблицы OPT(k, b) для хранения максимальных ценностей
    # Размер (n+1) x (capacity+1)
    opt = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Инициализация таблицы x(k, b) для отслеживания выбора предметов
    # x[k][b] = 1, если k-й предмет выбран в оптимальном решении (k, b)
    # x[k][b] = 0 в противном случае
    x = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Прямой ход алгоритма динамического программирования
    # Заполняем таблицы по формулам (1)-(4)
    
    for k in range(1, n + 1):
        for b in range(capacity + 1):
            v_k = volumes[k - 1]  # объём k-го предмета (индекс с 0)
            c_k = values[k - 1]   # ценность k-го предмета (индекс с 0)
            
            if k == 1:
                # Базовый случай: только первый предмет (формула 1 и 2)
                if v_k <= b:
                    opt[k][b] = c_k
                    x[k][b] = 1
                else:
                    opt[k][b] = 0
                    x[k][b] = 0
            else:
                # Рекуррентный случай: k >= 2 (формула 3 и 4)
                if v_k <= b:
                    # Случай 1: k-й предмет не выбран
                    value_without = opt[k - 1][b]
                    
                    # Случай 2: k-й предмет выбран
                    value_with = opt[k - 1][b - v_k] + c_k
                    
                    # Выбираем максимум (формула 3)
                    if value_with > value_without:
                        opt[k][b] = value_with
                        x[k][b] = 1
                    else:
                        opt[k][b] = value_without
                        x[k][b] = 0
                else:
                    # v_k > b: предмет не помещается
                    opt[k][b] = opt[k - 1][b]
                    x[k][b] = 0
    
    # Максимальная ценность находится в opt[n][capacity]
    max_value = opt[n][capacity]
    
    # Обратный ход: восстанавливаем выбранные предметы
    # Начинаем с k = n и остатка ёмкости b = capacity
    selected_items = []
    k = n
    remaining_capacity = capacity
    
    while k > 0:
        # Проверяем, был ли выбран k-й предмет
        # x[k][remaining_capacity] = x(k, b) из формулы (4)
        if x[k][remaining_capacity] == 1:
            # k-й предмет выбран (индекс в исходном списке: k-1)
            selected_items.append(k - 1)
            # Уменьшаем оставшуюся ёмкость на объём выбранного предмета
            remaining_capacity -= volumes[k - 1]
        
        k -= 1
    
    # Предметы добавлялись в обратном порядке
    selected_items.reverse()
    
    return max_value, selected_items


def main():
    # Пример 1: простой случай
    print("=" * 60)
    print("ПРИМЕР 1: Простой случай")
    print("=" * 60)
    
    volumes_1 = [2, 3, 4, 5]  # объёмы предметов
    values_1 = [3, 4, 5, 6]   # ценности предметов
    capacity_1 = 8            # вместимость рюкзака
    
    print(f"Предметы:          {list(range(len(volumes_1)))}")
    print(f"Объёмы:            {volumes_1}")
    print(f"Ценности:          {values_1}")
    print(f"Вместимость рюкзака: {capacity_1}")
    print()
    
    max_val_1, selected_1 = knapsack_solver(volumes_1, values_1, capacity_1)
    
    print(f"Максимальная ценность: {max_val_1}")
    print(f"Выбранные предметы (индексы): {selected_1}")
    total_volume_1 = sum(volumes_1[i] for i in selected_1)
    total_value_1 = sum(values_1[i] for i in selected_1)
    print(f"Суммарный объём: {total_volume_1}")
    print(f"Суммарная ценность: {total_value_1}")
    print()
    
    # Пример 2: более сложный случай
    print("=" * 60)
    print("ПРИМЕР 2: Более сложный случай")
    print("=" * 60)
    
    volumes_2 = [1, 2, 3, 4, 5]
    values_2 = [1, 6, 10, 13, 15]
    capacity_2 = 10
    
    print(f"Предметы:          {list(range(len(volumes_2)))}")
    print(f"Объёмы:            {volumes_2}")
    print(f"Ценности:          {values_2}")
    print(f"Вместимость рюкзака: {capacity_2}")
    print()
    
    max_val_2, selected_2 = knapsack_solver(volumes_2, values_2, capacity_2)
    
    print(f"Максимальная ценность: {max_val_2}")
    print(f"Выбранные предметы (индексы): {selected_2}")
    total_volume_2 = sum(volumes_2[i] for i in selected_2)
    total_value_2 = sum(values_2[i] for i in selected_2)
    print(f"Суммарный объём: {total_volume_2}")
    print(f"Суммарная ценность: {total_value_2}")
    print()
    
    # Пример 3: случай, когда в рюкзак ничего не помещается
    print("=" * 60)
    print("ПРИМЕР 3: Ограниченная ёмкость")
    print("=" * 60)
    
    volumes_3 = [5, 7, 3, 4]
    values_3 = [10, 15, 8, 12]
    capacity_3 = 6
    
    print(f"Предметы:          {list(range(len(volumes_3)))}")
    print(f"Объёмы:            {volumes_3}")
    print(f"Ценности:          {values_3}")
    print(f"Вместимость рюкзака: {capacity_3}")
    print()
    
    max_val_3, selected_3 = knapsack_solver(volumes_3, values_3, capacity_3)
    
    print(f"Максимальная ценность: {max_val_3}")
    print(f"Выбранные предметы (индексы): {selected_3}")
    if selected_3:
        total_volume_3 = sum(volumes_3[i] for i in selected_3)
        total_value_3 = sum(values_3[i] for i in selected_3)
        print(f"Суммарный объём: {total_volume_3}")
        print(f"Суммарная ценность: {total_value_3}")
    else:
        print("В рюкзак ничего не поместилось")
    print()


if __name__ == "__main__":
    main()