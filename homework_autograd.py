import torch
import math


def simple_autograd():
    # === Задание 2.1: Простые вычисления с градиентами ===
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


def mse_gradient():
    # === Задание 2.2: Градиент функции потерь MSE ===

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


def chain_rule_check():
    # === Задание 2.3: Цепное правило ===
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


if __name__ == "__main__":
    simple_autograd()
    mse_gradient()
    chain_rule_check()
