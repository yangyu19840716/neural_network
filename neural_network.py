# -*- coding:utf-8 -*-

import numpy as np

class neural_network(object):
    # L 层数，
    # N 大小为L + 1的数组 存放每层的数量, O为Input层,
    # A 激活函数 大小为 L 的数组 （是否可以按Node分?）
    # S 初始化缩放系数 大小为 L 的数组
    def __init__(self, L, N, A, S):
        L = L + 1
        self.L = L
        self.n = [N[0]] # L + 1, 存放每层的数量, 0层为Input层
        self.a = [None] # 激活函数 0层为None
        # z = W * x + b 特征函数
        self.W = [None] # 权重矩阵 0层为None
        self.b = [None] # 常量 0层为None
        # self.s = [None] # 初始化缩放系数 0层为None
        for l in xrange(1, L):
            i = l - 1 # A, S 的 index
            n = N[l]
            last_n = N[i]
            s = S[i]
            W = np.random.randn(n, last_n) * s
            b = np.zeros((n, 1))
            self.n.append(n)
            self.a.append(A[i])
            self.W.append(W)
            self.b.append(b)
            # self.s.append(s)

    def forward(self):
        pass

    def backward(self):
        pass