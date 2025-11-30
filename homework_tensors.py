import torch


def create_tensors():
    # === Задание 1.1: Создание тензоров ===

    # Тензор 3x4, случайные числа от 0 до 1
    t_rand = torch.rand(3, 4)
    print(f"Random 3x4:\n{t_rand}")

    # Тензор 2x3x4, заполненный нулями
    t_zeros = torch.zeros(2, 3, 4)
    print(f"\nZeros 2x3x4 shape: {t_zeros.shape}")

    # Тензор 5x5, заполненный единицами
    t_ones = torch.ones(5, 5)
    print(f"\nOnes 5x5:\n{t_ones}")

    # Тензор 4x4 с числами от 0 до 15 (используем arange и reshape)
    t_range = torch.arange(16).reshape(4, 4)
    print(f"\nRange 4x4:\n{t_range}")

    return t_rand, t_zeros, t_ones, t_range


def tensor_operations():
    # === Задание 1.2: Операции с тензорами ===

    A = torch.rand(3, 4)
    B = torch.rand(4, 3)
    print(f"Tensor A shape: {A.shape}, Tensor B shape: {B.shape}")

    # Транспонирование тензора A
    A_T = A.T
    print(f"Transposed A shape: {A_T.shape}")

    # Матричное умножение A и B (результат 3x3)
    matmul_AB = torch.matmul(A, B)  # или A @ B
    print(f"Matmul A @ B:\n{matmul_AB}")

    # Поэлементное умножение A и транспонированного B
    elementwise_mul = A * B.T
    print(f"Elementwise mul (A * B.T):\n{elementwise_mul}")

    # Сумма всех элементов тензора A
    sum_A = torch.sum(A)
    print(f"Sum of A: {sum_A.item()}")


def indexing_and_slicing():
    # === Задание 1.3: Индексация и срезы ===

    # Тензор 5x5x5
    T = torch.randn(5, 5, 5)

    # Первая строка (первая строка первой матрицы)
    first_row = T[0]
    print(f"First row (slice 0) shape: {first_row.shape}")

    # Последний столбец (по последней размерности)
    last_col = T[..., -1]
    print(f"Last column shape: {last_col.shape}")

    # Подматрица 2x2 из центра тензора
    center_submatrix = T[2, 1:3, 1:3]
    print(f"Center 2x2 submatrix:\n{center_submatrix}")

    # Все элементы с четными индексами (в выпрямленном виде)
    even_indices_elements = T.flatten()[::2]
    print(f"Elements with even indices count: {even_indices_elements.numel()}")


def shape_manipulation():
    # === Задание 1.4: Работа с формами ===

    t = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]

    for s in shapes:
        reshaped = t.reshape(s)
        print(f"Reshaped to {s}: {reshaped.shape}")


if __name__ == "__main__":
    create_tensors()
    tensor_operations()
    indexing_and_slicing()
    shape_manipulation()
