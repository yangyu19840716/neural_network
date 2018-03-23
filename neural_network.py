# -*- coding:utf-8 -*-

import numpy as np

def active_fun_1(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivative_active_fun_1(z):
    exp_z = np.exp(-z)
    g = exp_z / ((1 + exp_z) * (1 + exp_z))
    return g

class neural_network:
    # L 层数，
    # N 大小为L + 1的数组 存放每层的数量, O为Input层, 最后一层为最终输出
    # A 激活函数 大小为 L 的数组 （是否可以按Node分?）
    # A 激活函数导数 大小为 L 的数组 （是否可以按Node分?）
    # S 初始化缩放系数 大小为 L 的数组
    # ALPHA 学习率
    def __init__(self, L, N, A, D_A, S, ALPHA):
        L = L + 1
        self.l = L
        self.n = N # L + 1, 存放每层的数量, 0层为Input层, 最后一层为最终输出, n[-1] = 1
        self.alpha = ALPHA
        A.insert(0, None)
        D_A.insert(0, None)
        self.a = A # 激活函数 0层为None
        self.d_a = D_A # 激活函数导数 0层为None
        # z = W * x + b 特征函数
        # a = A(z) # A为激发函数 a 并不是self.a, 而是激发结果, 即当层计算的z, 或者下层计算的x
        self.W = [None] # 权重矩阵 0层为None
        self.b = [None] # 常量 0层为None
        self.x = [None] * L # Input
        self.z = [None] * L # Output
        S.insert(0, None)
        # self.s = S # 初始化缩放系数 0层为None
        for l in xrange(1, L):
            n = N[l]
            last_n = N[l - 1]
            W = np.random.randn(n, last_n) * S[l]
            b = np.zeros(n)
            self.W.append(W)
            self.b.append(b)

        # self.W = [   None,
        #         np.array([[1, 2, 3], [4, 5, 6]]),
        #         np.array([[1, 2], [3, 4], [5, 6]]),
        #         np.array([[1, 2, 3]])
        # ]

    def training_data(self, m, data = None):
        self.m = m # 样本组数

        # 数据 每一组 N[0] 个
        self.sample = data if data else np.random.randn(m, N[0] + 1) # 最后一个为结果

    # index 为 第几组 sample
    def forward(self, index):
        self.x[0] = self.sample[index][:-1]
        # 遍历层
        for i in xrange(1, self.l):
            z = (np.dot(self.W[i], self.x[i - 1]) + self.b[i])
            self.z[i] = z
            self.x[i] = self.a[i](z)

    def backward(self, index):
        y = self.sample[index][-1]
        a = self.z[-1] # 这里 a = hat_y
        dz = a - y
        for i in xrange(self.l - 1, 0, -1):
            # da = - y / a + (1 - y) / (1 - a)
            dW = dz * a / self.n[i]
            db = np.sum(dz) / self.n[i]
            a = self.z[i - 1]
            dz =  self.W[i].T * dz * self.d_a(a)
            self.W[i] -= self.alpha * dW
            self.b[i] -= self.alpha * db

    def lost(self, hat_y , y):
        # L(hat_y, y) = Lost(hat_y, y) = log(hat_y) + (1 - y) * log(1 - hat_y)
        # J(w, b) = Cast(w, b) = sum(L(hat_y, y), 1, m) / m
        return np.log(hat_y) + (1 + y) * np.log(1 - hat_y)

    def cast(self, m):
        pass

    def trainning(self):
        for i in xrange(self.m):
            self.forward(i)
            self.backward(i)

        print 'trainning end'

if __name__ == '__main__':
    import time

    L = 3
    N = [3, 2, 3, 1]
    A = [active_fun_1] * L
    D_A = [derivative_active_fun_1] * L
    S = [0.01] * L
    ALPHA = 0.01

    job = neural_network(L, N, A, D_A, S, ALPHA)
    job.training_data(100)
    # job.training_data(1, [[1, 2, 3, 4]])

    print 'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW'
    print job.W

    print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    print job.b

    t = time.time()
    print 'start', t

    job.trainning()

    tt = time.time()
    print 'end', tt
    print 'total', tt - t

    print 'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW'
    print job.W

    print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    print job.b

    print 'z =', job.z[-1]