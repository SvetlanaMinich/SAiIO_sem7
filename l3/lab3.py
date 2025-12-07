P = 3  
Q = 3 

# матрица прибыли A: строки - агенты, столбцы - ресурсы
A = [
    [0, 1, 2, 3],  # агент 1
    [0, 0, 1, 2],  # агент 2
    [0, 2, 2, 3]   # агент 3
]


B = [[0] * (Q + 1) for _ in range(P)]
C = [[0] * (Q + 1) for _ in range(P)]


for p in range(P):
    for q in range(Q + 1):
        if p == 0:  # один агент (может взять себе все ресурсы)
            B[p][q] = A[p][q]
            C[p][q] = q
        else:
            max_val = float('-inf')
            best_i = -1
            for i in range(q + 1):
                # делим q ресурсов между первыми p агентами
                # берем ресурс для агента p + ресурс для агентов p-1 (оставшиеся ресурсы после агента p) (определен максимум шагом ранее)
                val = A[p][i] + B[p - 1][q - i]
                if val > max_val:
                    max_val = val
                    best_i = i
            B[p][q] = max_val
            # сохраняем индекс ресурса, который отдадим этому агенту
            C[p][q] = best_i

max_profit = B[P - 1][Q]
print("Максимальная прибыль:", max_profit)


allocation = [0] * P
q_remaining = Q
p_current = P - 1
while p_current >= 0:
    alloc = C[p_current][q_remaining] 
    allocation[p_current] = alloc
    q_remaining -= alloc
    p_current -= 1

print("Распределение ресурсов по агентам:", allocation)

print("\nМатрица B:")
for row in B:
    print(row)

print("\nМатрица C:")
for row in C:
    print(row)
