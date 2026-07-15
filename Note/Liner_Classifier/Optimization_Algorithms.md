# 深度学习优化算法总结

这份笔记总结课程中常见的优化算法：SGD、SGD + Momentum、Nesterov、AdaGrad、RMSProp、Adam。它们的目标都是根据梯度更新参数，使损失函数逐步下降：

```text
theta <- theta - step
```

其中 `theta` 表示模型参数，`grad` 表示当前 mini-batch 上计算出的梯度。

## 总体对比

| Algorithm | 一阶动量 Momentum | 二阶矩 / 自适应学习率 | 二阶矩泄漏衰减 | 偏差修正 | 核心特点 |
| --- | --- | --- | --- | --- | --- |
| SGD | 否 | 否 | 否 | 否 | 简单稳定，但在病态曲率中容易震荡 |
| SGD + Momentum | 是 | 否 | 否 | 否 | 累积历史梯度方向，加速一致方向，抑制震荡 |
| Nesterov | 是 | 否 | 否 | 否 | 先“向前看”再算梯度，通常比普通 Momentum 更稳 |
| AdaGrad | 是/累积梯度平方 | 是 | 否 | 否 | 适合稀疏特征，但学习率会持续变小 |
| RMSProp | 是/指数平均梯度平方 | 是 | 是 | 否 | 修复 AdaGrad 学习率过早衰减的问题 |
| Adam | 是 | 是 | 是 | 是 | Momentum + RMSProp + 偏差修正，深度学习中最常用 |

> 注：表中“一阶动量”指跟踪梯度的一阶矩；“二阶矩”指跟踪梯度平方，用于给每个参数单独调整学习率。

## 1. SGD

### 核心公式

```text
theta = theta - learning_rate * grad
```

### 特征

- 优点：实现简单，内存开销小。
- 缺点：每一步只看当前梯度，方向容易受 mini-batch 噪声影响。
- 在某些方向曲率大、某些方向曲率小的损失地形中，SGD 容易在陡峭方向来回震荡，在平缓方向前进很慢。

### NumPy 实现

```python
def sgd_update(w, dw, learning_rate=1e-2):
    """
    w: 参数
    dw: 参数梯度
    """
    w = w - learning_rate * dw
    return w
```

### PyTorch 用法

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

## 2. SGD + Momentum

### 直觉

Momentum 会维护一个速度 `v`。如果连续几步梯度方向相近，速度会越来越大，从而加速前进；如果梯度方向来回变化，正负方向会相互抵消，从而减少震荡。

### 核心公式

常见写法：

```text
v = momentum * v - learning_rate * grad
theta = theta + v
```

课程中有时也写成：

```text
v = momentum * v + grad
theta = theta - learning_rate * v
```

两种写法本质类似，只是把学习率放在了不同位置。

### 特征

- 跟踪一阶矩，也就是梯度的指数滑动平均。
- 在峡谷形损失函数中，比普通 SGD 更快、更平滑。
- 需要额外保存一个和参数同形状的速度变量 `v`。
- 常用 `momentum = 0.9`。

### NumPy 实现

```python
def sgd_momentum_update(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.get("learning_rate", 1e-2)
    momentum = config.get("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = momentum * v - lr * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config
```

### PyTorch 用法

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
)
```

## 3. Nesterov Momentum

### 直觉

普通 Momentum 是在当前位置算梯度，然后用速度更新。Nesterov Momentum 会先根据当前速度“预判”下一步大概在哪里，再在这个预判位置计算梯度，因此有一种提前刹车的效果。

### 核心公式

```text
lookahead = theta + momentum * v
grad = gradient(lookahead)
v = momentum * v - learning_rate * grad
theta = theta + v
```

### 特征

- 也是一阶动量方法。
- 比普通 Momentum 更具有前瞻性。
- 在接近最优点时，通常能减少过冲。
- PyTorch 中通过 `nesterov=True` 开启，但必须同时设置 `momentum > 0`。

### NumPy 实现

```python
def nesterov_update(w, grad_fn, config=None):
    """
    grad_fn: 一个函数，输入参数 w，返回该位置的梯度。
    """
    if config is None:
        config = {}
    lr = config.get("learning_rate", 1e-2)
    momentum = config.get("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    lookahead_w = w + momentum * v
    dw = grad_fn(lookahead_w)

    v = momentum * v - lr * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config
```

### PyTorch 用法

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    nesterov=True,
)
```

## 4. AdaGrad

### 直觉

AdaGrad 会累计每个参数历史梯度的平方。某个参数经常出现大梯度，说明它已经被频繁更新，就降低它的学习率；某个参数很少出现梯度，就给它相对更大的学习率。

### 核心公式

```text
cache = cache + grad ** 2
theta = theta - learning_rate * grad / (sqrt(cache) + eps)
```

### 特征

- 为每个参数提供自适应学习率。
- 适合稀疏特征，例如 NLP 中的词向量。
- 缺点是 `cache` 单调递增，导致有效学习率越来越小，训练后期可能过早停滞。
- 常用 `eps = 1e-8`，防止除以 0。

### NumPy 实现

```python
def adagrad_update(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.get("learning_rate", 1e-2)
    eps = config.get("epsilon", 1e-8)
    cache = config.get("cache", np.zeros_like(w))

    cache = cache + dw ** 2
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    config["cache"] = cache
    return next_w, config
```

### PyTorch 用法

```python
optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=1e-2,
    eps=1e-10,
)
```

## 5. RMSProp

### 直觉

RMSProp 可以看成 AdaGrad 的改进版。它不再把所有历史梯度平方无衰减地累加，而是使用指数滑动平均，只重点关注最近一段时间的梯度平方。

### 核心公式

```text
cache = decay_rate * cache + (1 - decay_rate) * grad ** 2
theta = theta - learning_rate * grad / (sqrt(cache) + eps)
```

### 特征

- 跟踪二阶矩，也就是梯度平方的指数滑动平均。
- 有“泄漏”的二阶矩估计，旧梯度的影响会逐渐减弱。
- 修复 AdaGrad 学习率不断变小的问题。
- 常用 `decay_rate = 0.99` 或 `0.9`。

### NumPy 实现

```python
def rmsprop_update(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.get("learning_rate", 1e-2)
    decay_rate = config.get("decay_rate", 0.99)
    eps = config.get("epsilon", 1e-8)
    cache = config.get("cache", np.zeros_like(w))

    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    config["cache"] = cache
    return next_w, config
```

### PyTorch 用法

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=1e-3,
    alpha=0.99,
    eps=1e-8,
)
```

## 6. Adam

### 直觉

Adam 结合了 Momentum 和 RMSProp：

- Momentum 部分：用一阶矩 `m` 记录梯度方向的滑动平均。
- RMSProp 部分：用二阶矩 `v` 记录梯度平方的滑动平均。
- Bias correction：因为 `m` 和 `v` 初始为 0，训练早期会偏小，所以需要进行偏差修正。

### 核心公式

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad ** 2

m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)

theta = theta - learning_rate * m_hat / (sqrt(v_hat) + eps)
```

### 特征

- 同时跟踪一阶矩和二阶矩。
- 使用泄漏的二阶矩估计。
- 使用偏差修正，解决训练初期矩估计偏小的问题。
- 一般默认参数就很好用：`beta1 = 0.9`，`beta2 = 0.999`，`eps = 1e-8`。
- 在深度学习训练中非常常用，尤其适合快速得到不错的 baseline。

### NumPy 实现

```python
def adam_update(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.get("learning_rate", 1e-3)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.999)
    eps = config.get("epsilon", 1e-8)
    m = config.get("m", np.zeros_like(w))
    v = config.get("v", np.zeros_like(w))
    t = config.get("t", 0) + 1

    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    config["m"] = m
    config["v"] = v
    config["t"] = t
    return next_w, config
```

### PyTorch 用法

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

## 统一训练循环示例

无论使用哪种优化器，PyTorch 的训练流程基本一致：

```python
import torch

model = MyModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()

        scores = model(x)
        loss = criterion(scores, y)

        loss.backward()
        optimizer.step()
```

## 如何选择优化算法

| 场景 | 推荐 |
| --- | --- |
| 想要最简单、最可控的 baseline | SGD |
| SGD 震荡明显，下降慢 | SGD + Momentum |
| 想在 Momentum 基础上更稳定 | Nesterov |
| 稀疏特征，参数更新频率差异很大 | AdaGrad |
| 想要自适应学习率，但不希望学习率无限衰减 | RMSProp |
| 大多数深度学习任务的默认选择 | Adam |
| 最终追求泛化性能，愿意调参 | SGD + Momentum 或 AdamW |

## 常见超参数

| Optimizer | 常用 learning rate | 其他常用参数 |
| --- | --- | --- |
| SGD | `1e-1`, `1e-2`, `1e-3` | 无 |
| SGD + Momentum | `1e-2`, `1e-3` | `momentum=0.9` |
| Nesterov | `1e-2`, `1e-3` | `momentum=0.9`, `nesterov=True` |
| AdaGrad | `1e-2` | `eps=1e-8` |
| RMSProp | `1e-3` | `alpha=0.99`, `eps=1e-8` |
| Adam | `1e-3`, `3e-4`, `1e-4` | `betas=(0.9, 0.999)` |

## 一句话记忆

- SGD：只看当前梯度，直接走。
- Momentum：带速度走，方向一致就加速。
- Nesterov：先向前看一步，再决定怎么走。
- AdaGrad：梯度大的参数越走越慢。
- RMSProp：只记近期梯度平方，避免 AdaGrad 后期走不动。
- Adam：Momentum + RMSProp + 偏差修正。

## Regularization

$L(W)=\sum_{i=1}^{N}L_i(f(w,x_i),y_i)+\lambda R(W)$
