import os
import sys

sys.path.append(os.pardir)
import numpy as np

from common.functions import cross_entropy_error, sigmoid, softmax
from common.gradient import numerical_gradient


class TwoLayerNet:
    # input_size: 入力層のニューロン数
    # hidden_size: 隠れ層のニューロン数
    # output_size: 出力層のニューロン数（クラス数）
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = x @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)  # 重みと入力・バイアスの行列計算
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 入力データ, t: 教師データ
    def numerical_gradient(self, x, t):
        # 現在の入力と教師データに対する損失関数の値を返す関数を定義
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # W1の勾配を求める
        # W1は785x50の行列なので、numerica
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print(net.params["W1"].shape)
print(net.params["b1"].shape)
print(net.params["W2"].shape)
print(net.params["b2"].shape)
