import numpy as np 
from scipy.optimize import linprog 
import math 

class BranchAndBound:
    def __init__(self, c, A, b, d_low, d_high):
        self.original_c = np.array(c, dtype=float)
        self.original_A = np.array(A, dtype=float) 
        self.original_b = np.array(b, dtype=float) 
        self.original_d_low = np.array(d_low, dtype=float)
        self.original_d_high = np.array(d_high, dtype=float)

        self.c_transformed = self.original_c.copy()
        self.A_transformed = self.original_A.copy()
        self.d_low_transformed = self.original_d_low.copy()
        self.d_high_transformed = self.original_d_high.copy()
        
        self.n = len(c) 
        self.m = len(b) 
        
        # Шаг 1: Преобразуем задачу таким образом, чтобы c <= 0
        self.sign_changes = []
        for i in range(self.n):
            if self.original_c[i] > 0:
                # Умножаем i-тую компоненту на -1
                self.c_transformed[i] = -self.original_c[i]
                # Умножаем i-тый столбец в матрице на -1
                self.A_transformed[:, i] = -self.A_transformed[:, i]
                # Умножаем i-тую компоненту нижних и верхних границ на -1 и меняем их местами
                low, high = self.original_d_low[i], self.original_d_high[i]
                self.d_low_transformed[i] = -high
                self.d_high_transformed[i] = -low
                self.sign_changes.append(i) 
        
        # Шаг 2: Отбросим условия целочисленности и приведем полученную задачу линейного программирования к канонической форме
        # m для каждого правого ограничения
        # 2*n для верхних и нижних (наши текущие значения A) ограничений
        self.total_vars = 2 * self.n + self.m 
        
        # Расширяем вектор коэффициентов целевой функции, где первые n элементов - преобразованные c
        self.c_ext = np.zeros(self.total_vars) 
        self.c_ext[:self.n] = self.c_transformed
        
        # Расширяем матрицу ограничений
        self.A_ext = np.zeros((self.m + self.n, self.total_vars))
        
        # Заполняем A_ext:
        # Верхний левый блок - преобразованная матрица A
        self.A_ext[:self.m, :self.n] = self.A_transformed
        # Блок для дополнительных переменных через единичные матрицы
        self.A_ext[:self.m, self.n : self.n + self.m] = np.eye(self.m)
        
        self.A_ext[self.m : , :self.n] = np.eye(self.n)
        self.A_ext[self.m : , self.n + self.m : ] = np.eye(self.n) 
        
        # Расширяем вектор правых частей ограничений
        self.b_ext = np.concatenate([self.original_b, self.d_high_transformed])
        
        # Расширяем вектор нижних границ, где первые n элементов - преобразованные d_low
        self.d_low_ext = np.zeros(self.total_vars)
        self.d_low_ext[:self.n] = self.d_low_transformed 
        
        # Шаг 3: Переменные + пустой стек
        self.best_solution = None # Лучшее найденное целочисленное решение
        self.best_value = -np.inf # Значение целевой функции для лучшего целочисленного решения 
        self.stack = []
        
    # Метод для решения канон задачи линейного программирования
    def solve_relaxed(self, delta, current_b):
        # delta - смещения для приведения нижней границы в 0 
        # alpha_prime - поправка для целевой функции, учитывающая смещение переменных (delta)
        alpha_prime = np.dot(self.c_ext, delta)
        # b_prime - модифицированный вектор правых частей ограничений
        b_prime = current_b - self.A_ext @ delta
        
        # Решаем задачу линейного программирования с помощью linprog
        # -self.c_ext, потому что linprog минимизирует, а мы хотим максимизировать self.c_ext
        # bounds - границы для переменных (от 0 до бесконечности)
        result = linprog(-self.c_ext, A_eq=self.A_ext, b_eq=b_prime, bounds=[(0, None)] * self.total_vars)
        
        if not result.success:
            # Задача не имеет допустимых решений
            return None, -np.inf, None
        
        solution = result.x 
        # value - значение целевой функции с учетом поправки
        value = np.dot(self.c_ext, solution) + alpha_prime
        
        return solution, value, b_prime 
    
    # Метод для проверки, являются ли переменные целочисленными
    def is_integer(self, x, tolerance=1e-6):
        # Проверяем первые n переменных (исходные переменные задачи)
        # abs(x[i] - round(x[i])) < tolerance - проверка на то, что число очень близко к целому
        return all(abs(x[i] - round(x[i])) < tolerance for i in range(self.n))
    
    
    def branch_and_bound(self):
        initial_delta = self.d_low_ext.copy() # Начальное смещение (нижние границы)
        initial_current_b = self.b_ext.copy() # Начальный вектор правых частей ограничений
        self.stack.append((initial_delta, initial_current_b)) 
        
        iteration = 0 
        while self.stack: 
            iteration += 1
            print(f"\nИтерация {iteration}")
            
            # Шаг 4: Извлекаем подзадачу из стека
            delta, current_b = self.stack.pop()
            
            # Решаем канон задачу для текущей подзадачи
            solution, value, b_prime = self.solve_relaxed(delta, current_b)
            
            if solution is None:
                print("Недопустимое решение") 
                continue
            
            print(f"Решение: {solution[:self.n]}, значение целевой функции: {value:.2f}")
            
            if self.is_integer(solution):
                if value > self.best_value: 
                    self.best_solution = solution + delta 
                    self.best_value = value 
                    print(f"Новое лучшее значение: {value:.2f}")
                continue 
            
            # Берем максимальное целое решение и если оно хуже текущего, то нет смысла
            # продолжать эту ветку (мы ищем лучшее решение)
            if math.floor(value) <= self.best_value: # Исправлено: floor
                print("Граница хуже - отсекаем")
                continue
            
            # Выбираем переменную для ветвления (ту, которая не является целой)
            branching_var = None
            for i in range(self.n):
                if abs(solution[i] - round(solution[i])) > 1e-6: 
                    branching_var = i 
                    break
            
            print(f"Ветвление по переменной x{branching_var + 1} = {solution[branching_var]:.2f}")
            
            frac = solution[branching_var] # Нецелое значение переменной
            # Округляем ее в меньшую и большую стороны
            floor_v = math.floor(frac) # Нижняя 
            ceil_v = math.ceil(frac) # Верхняя 
            
            # Порождаем новые подзадачи и записываем их в стэк
            j = self.m + branching_var 
            adjust = floor_v - b_prime[j] # Вычисляем корректировку
            new_current_b_left = current_b.copy()
            new_current_b_left[j] += adjust # Применяем корректировку
            new_delta_left = delta.copy() # Копируем текущие смещения
            self.stack.append((new_delta_left, new_current_b_left)) 
            
            new_delta_right = delta.copy()
            new_delta_right[branching_var] += ceil_v # Увеличиваем нижнюю границу для переменной ветвления
            new_current_b_right = current_b.copy() # Копируем текущий b
            self.stack.append((new_delta_right, new_current_b_right))
            
            print(f"Созданы подзадачи: x{branching_var + 1} <= {floor_v} (через b) и x{branching_var + 1} >= {ceil_v} (через delta)")
        
        if self.best_solution is None:
            return None, -np.inf 
        
        # Берем первые n переменных (мы добавили вспомогательные переменные и теперь они нам не нужны)
        original_solution = self.best_solution[:self.n].copy()
        for i in self.sign_changes:
            # Отменяем изменения знаков
            original_solution[i] = -original_solution[i] 
        
        return original_solution 

# Пример использования
# c = [1, 1] - коэффициенты целевой функции, к чему мы стремимся
# A = [[5, 9], [9, 5]] - коэффициенты ограничений 
# b = [63, 63] - правые части ограничений 
# d_low = [1, 1] - нижние границы для переменных 
# d_high = [6, 6] - верхние границы для переменных
c = [1, 1]
A = [[5, 9], [9, 5]]
b = [63, 63]
d_low = [1, 1]
d_high = [6, 6]

solver = BranchAndBound(c, A, b, d_low, d_high)
solution = solver.branch_and_bound()

print("\n" + "="*50)
if solution is not None:
    print(f"Оптимальное решение: x1 = {solution[0]}, x2 = {solution[1]}")
else:
    print("Проблема не имеет допустимых решений")