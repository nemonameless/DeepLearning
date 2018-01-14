# encoding=utf-8
import numpy as np


class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x)  # 使用随机数初始化参数
        self.bias = np.random.randn(1)  # 使用随机数初始化参数

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.dot(self.weights, x) + self.bias  # 计算w11*a1+w12*a2+bias1
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        self.dw = d * self.x  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.db = d
        self.dx = d * self.weights
        return self.dw, self.db  # 返回求得的参数梯度，注意这里如果要继续反向传递梯度，应该返回self.dx


class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self):  # 这里sigmoid是最后一层，所以从这里开始反向计算梯度
        sig = self.sigmoid(self.x)
        self.dx = sig * (1 - sig)
        return self.dx  # 反向传递梯度


def main():
    fc = FullyConnect(2, 1)
    sigmoid = Sigmoid()
    x = np.array([[1], [2]])
    print 'parameters: weights:', fc.weights, ' bias:', fc.bias, ' input: ', x

    # 执行前向计算
    y1 = fc.forward(x)
    y2 = sigmoid.forward(y1)
    print 'forward result: ', y2

    # 执行反向传播
    d1 = sigmoid.backward()
    dx = fc.backward(d1)
    print 'backward result: ', dx


if __name__ == '__main__':
    main()
