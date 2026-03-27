import numpy as np


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4  # 小さい値
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]  # 3.0

        # f(x+h)
        x[idx] = tmp_val + h  # 3.0001
        fxh1 = f(x)  # 3.0001**2 + 4.0**2

        # f(x-h)
        x[idx] = tmp_val - h  # 2.9999
        fxh2 = f(x)  # 2.9999**2 + 4.0**2

        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 微分
        x[idx] = tmp_val  # 元に戻して、4.0の計算に影響を与えないようにする

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x  # 初期のランダムな値

    # 指定ステップ分勾配法を繰り返す
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad  # 学習率を掛けて、更新

    return x


print(gradient_descent(function_2, init_x=np.array([-3.0, 4.0]), lr=10.0, step_num=100))
print(
    gradient_descent(function_2, init_x=np.array([-3.0, 4.0]), lr=1e-10, step_num=100)
)
