# Отчет по домашнему заданию 1

### 1.1 Создание тензоров 
### Решение
```python
def create_tensors():
    # Задание 1.1: Создание тензоров

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
```
### *Вывод из консоли*
```
Random 3x4:
tensor([[0.5181, 0.6770, 0.8749, 0.5079],
        [0.6439, 0.8545, 0.9530, 0.7566],
        [0.2512, 0.1567, 0.4873, 0.4969]])

Zeros 2x3x4 shape: torch.Size([2, 3, 4])

Ones 5x5:
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])

Range 4x4:
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
``` 

### 1.2 Операции с тензорами 
### Решение
```python
def tensor_operations():
    # Задание 1.2: Операции с тензорами

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
```
### *Вывод из консоли*
```
Tensor A shape: torch.Size([3, 4]), Tensor B shape: torch.Size([4, 3])
Transposed A shape: torch.Size([4, 3])
Matmul A @ B:
tensor([[0.6813, 0.7057, 0.3872],
        [0.2678, 0.2964, 0.3405],
        [0.6975, 0.9200, 0.8738]])
Elementwise mul (A * B.T):
tensor([[4.7926e-01, 5.6929e-04, 1.3206e-01, 6.9416e-02],
        [9.9171e-02, 5.5178e-02, 4.9162e-02, 9.2902e-02],
        [2.0548e-02, 1.0214e-01, 6.5903e-01, 9.2090e-02]])
Sum of A: 4.0436906814575195
``` 


### 1.3 Индексация и срезы
### Решение
```python
def indexing_and_slicing():
    # Задание 1.3: Индексация и срезы
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
```
### *Вывод из консоли*
```
First row (slice 0) shape: torch.Size([5, 5])
Last column shape: torch.Size([5, 5])
Center 2x2 submatrix:
tensor([[ 1.2602,  0.1085],
        [ 0.8067, -1.7298]])
Elements with even indices count: 63
``` 

### 1.4 Работа с формами 
### Решение
```python
def shape_manipulation():
    # Задание 1.4: Работа с формами

    t = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]

    for s in shapes:
        reshaped = t.reshape(s)
        print(f"Reshaped to {s}: {reshaped.shape}")
```
### *Вывод из консоли*
```
Reshaped to (2, 12): torch.Size([2, 12])
Reshaped to (3, 8): torch.Size([3, 8])
Reshaped to (4, 6): torch.Size([4, 6])
Reshaped to (2, 3, 4): torch.Size([2, 3, 4])
Reshaped to (2, 2, 2, 3): torch.Size([2, 2, 2, 3])
``` 

## Задание 2: Автоматическое дифференцирование 



### 2.1 Простые вычисления с градиентами 
### Решение
```python
def simple_autograd():
    print('-------------')
    # Задание 2.1: Простые вычисления с градиентами
    # f(x,y,z) = x^2 + y^2 + z^2 + 2xyz

    # Создаем тензоры
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # Функция
    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
    f.backward()

    print(f"f(x,y,z) = {f.item()}")
    print(f"df/dx (autograd): {x.grad.item()}")
    print(f"df/dy (autograd): {y.grad.item()}")
    print(f"df/dz (autograd): {z.grad.item()}")

    # Аналитическая проверка
    analytic_dx = 2 * x.item() + 2 * y.item() * z.item()
    print(f"df/dx (analytic): {analytic_dx}")
    assert math.isclose(x.grad.item(), analytic_dx), "Ошибка в градиенте x!"
```
### *Вывод из консоли*
```
f(x,y,z) = 77.0
df/dx (autograd): 28.0
df/dy (autograd): 22.0
df/dz (autograd): 20.0
df/dx (analytic): 28.0
``` 

### 2.2 Градиент функции потерь 
### Решение
```python
def mse_gradient():
    #  Задание 2.2: Градиент функции потерь MSE
    # Данные
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])

    # Веса (инициализация)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Предсказание и ошибка
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    print(f"Loss: {loss.item()}")
    print(f"dL/dw: {w.grad.item()}")
    print(f"dL/db: {b.grad.item()}")
```
### *Вывод из консоли*
```
Loss: 7.5
dL/dw: -15.0
dL/db: -5.0
``` 

### 2.3 Цепное правило 
### Решение
```python
def chain_rule_check():
    # Задание 2.3: Цепное правило
    # f(x) = sin(x^2 + 1)

    x = torch.tensor(1.5, requires_grad=True)

    # Функция
    f = torch.sin(x ** 2 + 1)

    # 1. Через backward
    f.backward(retain_graph=True)
    grad_backward = x.grad.item()

    # 2. Через torch.autograd.grad
    x.grad.zero_()
    grad_calc = torch.autograd.grad(f, x)[0].item()

    print(f"f(x) value: {f.item()}")
    print(f"df/dx via .backward(): {grad_backward}")
    print(f"df/dx via autograd.grad: {grad_calc}")

    # Аналитически: f' = cos(x^2 + 1) * 2x
    analytic = math.cos(1.5 ** 2 + 1) * 2 * 1.5
    print(f"df/dx analytic: {analytic}")
```
### *Вывод из консоли*
```
f(x) value: -0.10819513350725174
df/dx via .backward(): -2.982388973236084
df/dx via autograd.grad: -2.982388973236084
df/dx analytic: -2.9823890282416388
``` 

## Задание 3: Сравнение производительности CPU vs CUDA 


### 3.1 Подготовка данных 
### Решение
```python
def create_large_matrices():
    # Задание 3.1: Подготовка данных

    sizes = [
        (64, 1024, 1024),
        (128, 512, 512),
        (256, 256, 256)
    ]
    data = []
    for s in sizes:
        t = torch.randn(*s)
        data.append(t)
    return data
```


### 3.2 Функция измерения времени 
### Решение
```python
def measure_time(func, device_type='cpu', runs=10):
    # Задание 3.2: Функция измерения времени

    if device_type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(3):
            func()
        torch.cuda.synchronize()

        start.record()
        for _ in range(runs):
            func()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / runs

    else:
        # CPU Warmup
        for _ in range(3):
            func()

        start = time.time()
        for _ in range(runs):
            func()
        end = time.time()
        return ((end - start) * 1000) / runs
```


### 3.3 Сравнение операций 
### Решение
```python
def compare_operations():
    # Задание 3.3: Сравнение операций CPU vs CUDA

    size = 2048

    # Данные
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    has_gpu = torch.cuda.is_available()

    if has_gpu:
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
    else:
        print("CUDA недоступна. Тестирование только на CPU.")
        a_gpu = b_gpu = None

    operations = {
        "Матричное умножение": {
            "cpu": lambda: torch.matmul(a_cpu, b_cpu),
            "gpu": lambda: torch.matmul(a_gpu, b_gpu)
        },
        "Сложение": {
            "cpu": lambda: a_cpu + b_cpu,
            "gpu": lambda: a_gpu + b_gpu
        },
        "Умножение (поэлементное)": {
            "cpu": lambda: a_cpu * b_cpu,
            "gpu": lambda: a_gpu * b_gpu
        },
        "Транспонирование": {
            "cpu": lambda: a_cpu.transpose(0, 1),
            "gpu": lambda: a_gpu.transpose(0, 1).contiguous()
        },
        "Сумма": {
            "cpu": lambda: torch.sum(a_cpu),
            "gpu": lambda: torch.sum(a_gpu)
        }
    }

    print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
    print("-" * 65)

    for name, ops in operations.items():
        t_cpu = measure_time(ops["cpu"], 'cpu')

        if has_gpu:
            t_gpu = measure_time(ops["gpu"], 'cuda')
            speedup = t_cpu / t_gpu if t_gpu > 0 else 0
            print(f"{name:<25} | {t_cpu:<10.2f} | {t_gpu:<10.2f} | {speedup:<10.1f}")
        else:
            print(f"{name:<25} | {t_cpu:<10.2f} | {'N/A':<10} | {'N/A':<10}")
```
### *Вывод из консоли*
```
Операция                  | CPU (мс)   | GPU (мс)   | Ускорение 
-----------------------------------------------------------------
Матричное умножение       | 26.67      | 1.02       | 26.2      
Сложение                  | 0.62       | 0.11       | 5.7       
Умножение (поэлементное)  | 0.60       | 0.11       | 5.4       
Транспонирование          | 0.00       | 0.12       | 0.0       
Сумма                     | 0.10       | 0.02       | 5.1 
``` 


### 3.4 Анализ результатов 
```
1. Матричное умножение получает наибольшее ускорение на GPU.
2. Операции, ограниченные памятью (сложение), имеют меньший прирост.
3. На малых матрицах GPU может быть медленнее из-за накладных расходов.
4. Передача данных между CPU и GPU - узкое место.
```