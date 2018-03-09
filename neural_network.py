# -*- coding:utf-8 -*-

import numpy as np

class neural_network:
    # L 层数，
    # N 大小为L + 1的数组 存放每层的数量, O为Input层,
    # A 激活函数 大小为 L 的数组 （是否可以按Node分?）
    # S 初始化缩放系数 大小为 L 的数组
    def __init__(self, L, N, A, S):
        L = L + 1
        self.l = L
        self.n = N # L + 1, 存放每层的数量, 0层为Input层
        A.insert(0, None)
        self.a = A # 激活函数 0层为None
        # z = W * x + b 特征函数
        self.W = [None] # 权重矩阵 0层为None
        self.b = [None] # 常量 0层为None
        self.x = [None] * L # Input
        self.z = [None] * L # Output
        # self.s = [None] # 初始化缩放系数 0层为None
        for l in xrange(1, L):
            i = l - 1 # A, S 的 index
            n = N[l]
            last_n = N[i]
            s = S[i]
            W = np.random.randn(n, last_n) * s
            b = np.zeros((n, 1))
            self.W.append(W)
            self.b.append(b)
            # self.s.append(s)

        # 数据 每一组 N[0] 个
        self.n_sample = 1000 # 样本组数
        self.sample = np.random.randn(n_sample, N[0])

    # index 为 第几组 sample
    def forward(self, index):
        self.x[0] = self.sample[index]
        # 遍历层
        for i in xrang(1, self.l):
            self.z[i] = self.W[i] * self.x[i - 1] + self.b
            self.x[i] = self.a[i](self.z[i])

    def backward(self):
        pass

    def tranning(self):
        for i in xrange(self.n_sample):
            self.forward(i)
            self.backward()

        return self.z[-1]