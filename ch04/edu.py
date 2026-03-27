import os
import sys

sys.path.append(os.pardir)
import numpy as np

from common.functions import softmax

# 入力データとラベルと重み
# 損失関数に対する勾配を求める

# 入力データ
# 仮の入力データ。実際は28x28の画像データなどを入れる
x = np.array([0.6, 0.9])
# ラベル
# クラス2が正解。数字の2のデータとする
t = np.array([0, 0, 1])
# 重み
W = np.random.rand(2, 3)

# 入力と重みの積を計算する
z1 = x @ W  # [0.77824779 0.47354438 0.85617466]

# ソフトマックス関数で確率に変換する
a1 = softmax(z1)  # [0.35481305 0.26161857 0.38356838]

# 交差エントロピー誤差を計算する
batch_size = a1.shape[0]
loss = -np.sum(np.log(a1[np.arange(batch_size), t.argmax(axis=1)] + 1e-7)) / batch_size
print(loss)
