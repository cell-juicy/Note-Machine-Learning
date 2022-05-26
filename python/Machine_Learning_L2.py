from functools import partial
from time import time
from scipy import interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# 定义代价函数J
def J_fun(t_0, t_1, _x_data, _y_data):
    # 若数据维度不同则报错
    if len(_x_data) != len(_y_data):
        raise ValueError('The dimension of X_data is different from that of Y_data')
    _result = 0
    for _i in range(len(_x_data)):
        _result += ((t_0 + t_1 * _x_data[_i]) - _y_data[_i])**2
    _result *= 1 / (2 * len(_x_data))
    return _result


# 定义梯度下降法用的两个迭代函数J_0与J_1与迭代过程loop
def J0_fun(t_0, t_1, _x_data, _y_data):
    # 若数据维度不同则报错
    if len(_x_data) != len(_y_data):
        raise ValueError('The dimension of X_data is different from that of Y_data')
    _result = 0
    for _i in range(len(_x_data)):
        _result += (t_0 + t_1 * _x_data[_i]) - _y_data[_i]
    _result *= 1 / len(_x_data)
    return _result


def J1_fun(t_0, t_1, _x_data, _y_data):
    # 若数据维度不同则报错
    if len(_x_data) != len(_y_data):
        raise ValueError('The dimension of X_data is different from that of Y_data')
    _result = 0
    for _i in range(len(_x_data)):
        _result += _x_data[_i] * ((t_0 + t_1 * _x_data[_i]) - _y_data[_i])
    _result *= 1 / len(_x_data)
    return _result


def loop_fun(t_0, t_1, alpha=0.001, j_list=()):
    # 检查是否传入J偏导函数
    for i in j_list:
        if not callable(i):
            raise ValueError('Please input J-function')
        if len(j_list) != 2:
            raise ValueError('Please input two J-function')
    _J0 = j_list[0]
    _J1 = j_list[1]
    _t0 = t_0 - alpha * _J0(t_0, t_1)
    _t1 = t_1 - alpha * _J1(t_0, t_1)
    return _t0, _t1


'''
创建数据集,以及一些前置的操作
'''
# 数据集创建
np.random.seed(10)
price = [0.36 * i + 2 * np.random.random() - 1 for i in range(2, 50)]
num = [0.24 * i + 5 * np.random.random() + 1.5 for i in range(2, 50)]
# 三个函数
J = partial(J_fun, _x_data=num, _y_data=price)
dJ0 = partial(J0_fun, _x_data=num, _y_data=price)
dJ1 = partial(J1_fun, _x_data=num, _y_data=price)
loop = partial(loop_fun, j_list=(dJ0, dJ1))
# matplotlib支持中文的处理
mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


# 可视化数据集
plt.scatter(num, price, c='r', s=15)
plt.xlabel('影响因素x')
plt.ylabel('价格y')
plt.title('训练集展示')
# plt.savefig('Figure_1.svg')
plt.show()
# 可视化代价函数
x, y = np.linspace(-40, 40, 1000), np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
Z = J(X, Y)
cground = plt.contourf(X, Y, Z, [0, 10, 40, 200, 500, 3000, 10000], cmap='hot_r')
cline = plt.contour(X, Y, Z, [10, 40, 200, 500, 3000], colors='k')
plt.clabel(cline, fontsize=7)
plt.colorbar(cground)
plt.title('代价函数J等高线图')
plt.xlabel('θ_0')
plt.ylabel('θ_1')
# plt.savefig('Figure_2.svg')
plt.show()

'''
迭代,记录迭代过程
'''
t0, t1 = -10, -2        # 迭代起点
t0_list = [t0]          # 记录集合
t1_list = [t1]
t0l, t1l = 0, 0         # 迭代中间量
t_start = time()
while (t0-t0l)**2+(t1-t1l)**2 > 0.00000001:
    t0l, t1l = t0, t1
    t0, t1 = loop(t0, t1, alpha=0.005)
    t0_list.append(t0)
    t1_list.append(t1)
t_end = time()


'''
可视化
'''
# 在代价函数J等高线图上显示迭代的过程
x, y = np.linspace(-40, 40, 1000), np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
Z = J(X, Y)
cground = plt.contourf(X, Y, Z, [0, 10, 40, 200, 500, 3000, 10000], cmap='hot_r')
cline = plt.contour(X, Y, Z, [10, 40, 200, 500, 3000], colors='k')
plt.clabel(cline, fontsize=7)
plt.colorbar(cground)
plt.xlabel('θ_0')
plt.ylabel('θ_1')
plt.title('迭代过程可视化')
plt.scatter(t0_list, t1_list, s=3, c=np.arange(len(t0_list))/len(t0_list))
# plt.savefig('Figure_3.svg')
plt.show()
# 在训练集散点图上显示最终成果
x = np.linspace(1, 19, 300)
line1, = plt.plot(x, t0 + t1 * x, c='k')
line2, = plt.plot(x, t0_list[3000] + t1_list[3000] * x, c='m')
line3, = plt.plot(x, t0_list[100] + t1_list[100] * x, c='y')
line4, = plt.plot(x, t0_list[0] + t1_list[0] * x, c='b')
plt.legend([line1, line2, line3, line4], ['最终结果', '3000次迭代', '100次迭代', '初始值'], loc='best')
plt.scatter(num, price, c='r', s=15)
plt.title('梯度下降法结果显示')
plt.xlabel('影响因素x')
plt.ylabel('价格y')
# plt.savefig('Figure_4.svg')
plt.show()
# 绘制代价函数随迭代次数的变化
plt.title('代价函数与迭代次数关系曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价函数J')
X = np.arange(101)
Y = np.array([J(t0_list[i], t1_list[i]) for i in X])
fun = interpolate.interp1d(X, Y, kind='cubic')
x = np.linspace(0, 40, 1000)
y = fun(x)
plt.plot(x, y)
# plt.savefig('Figure_5.svg')
plt.show()
# 训练结果打印为报告
print(f'迭代次数:{len(t0_list)-1}\n用时:{t_end-t_start}s\n迭代最终解: θ0 = {t0}\n          θ1 = {t1}')
