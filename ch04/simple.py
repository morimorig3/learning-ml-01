import os
import sys

sys.path.append(os.pardir)
import numpy as np

from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 正規分布からランダムに重みを初期化する

    # 行列の積を計算する
    def predict(self, x):
        return x @ self.W

    def loss(self, x, t):
        # 重みと入力の積を計算
        z = self.predict(x)

        # ソフトマックス関数で確率に変換する
        y = softmax(z)

        # 交差エントロピー誤差を計算する
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

t = np.array([0, 0, 1])
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
