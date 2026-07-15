import numpy as np
import matplotlib.pyplot as plt

# 1. 定义一个具有“病态曲率”（峡谷地形）的损失函数
# f(x, y) = 0.1 * x^2 + 2 * y^2
# 在 x 方向极其平缓，在 y 方向极其陡峭
def loss_function(x, y):
    return 0.1 * x**2 + 2 * y**2

# 计算该函数的梯度
def gradient(x, y):
    dx = 0.2 * x
    dy = 4.0 * y
    return np.array([dx, dy])

# 2. 初始化超参数
learning_rate = 0.4
momentum_decay = 0.9
num_steps = 30
start_pos = np.array([-8.0, 2.0]) # 起点位置

# 3. 模拟纯 SGD 的轨迹
sgd_pos = start_pos.copy()
sgd_path = [sgd_pos.copy()]

for _ in range(num_steps):
    grad = gradient(sgd_pos[0], sgd_pos[1])
    sgd_pos -= learning_rate * grad
    sgd_path.append(sgd_pos.copy())

# 4. 模拟带动量 (Momentum) 的 SGD 轨迹
mom_pos = start_pos.copy()
mom_path = [mom_pos.copy()]
velocity = np.array([0.0, 0.0]) # 初始速度为 0

for _ in range(num_steps):
    grad = gradient(mom_pos[0], mom_pos[1])
    # 动量更新公式
    velocity = momentum_decay * velocity + grad
    mom_pos -= learning_rate * velocity
    mom_path.append(mom_pos.copy())

# 转换为 NumPy 数组方便画图
sgd_path = np.array(sgd_path)
mom_path = np.array(mom_path)

# 5. 开始绘制可视化图表
plt.figure(figsize=(10, 6))

# 绘制峡谷的等高线
x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-5, 5, 100))
z_grid = loss_function(x_grid, y_grid)
plt.contour(x_grid, y_grid, z_grid, levels=np.logspace(-0.5, 3, 20), cmap='gray', alpha=0.5)

# 绘制两条轨迹
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r-o', label='Standard SGD (Jittering)', markersize=4)
plt.plot(mom_path[:, 0], mom_path[:, 1], 'b-o', label='SGD with Momentum (Smooth & Fast)', markersize=4)

# 标记起点和终点（全局最优解在 [0, 0]）
plt.plot(start_pos[0], start_pos[1], 'k*', markersize=15, label='Start')
plt.plot(0, 0, 'y*', markersize=15, label='Global Minimum (0,0)')

plt.title('SGD vs SGD with Momentum in a "Ravine"')
plt.xlabel('X (Shallow Dimension)')
plt.ylabel('Y (Steep Dimension)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()