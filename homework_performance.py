import torch
import time


def create_large_matrices():
    # === Задание 3.1: Подготовка данных ===

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


def measure_time(func, device_type='cpu', runs=10):
    # === Задание 3.2: Функция измерения времени ===

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


def compare_operations():
    # === Задание 3.3: Сравнение операций CPU vs CUDA ===

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

    # Заголовок таблицы (оставляем, так как это результат)
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


if __name__ == "__main__":
    create_large_matrices()
    compare_operations()
    print(
    "\nАнализ результатов:",
    "1. Матричное умножение получает наибольшее ускорение на GPU.",
    "2. Операции, ограниченные памятью (сложение), имеют меньший прирост.",
    "3. На малых матрицах GPU может быть медленнее из-за накладных расходов.",
    "4. Передача данных между CPU и GPU - узкое место.",
    sep='\n'
)
