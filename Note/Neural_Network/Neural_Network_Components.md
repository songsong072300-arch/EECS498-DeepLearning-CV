 # 神经网络主要组件总结

这份笔记总结构建神经网络时常见的几个部分：全连接层、激活函数、卷积层、池化层、归一化层，以及它们在经典卷积网络中的组合方式。

一个典型 CNN 可以写成：

```text
[Conv, Activation, Pool] x N -> Flatten -> [FC, Activation] x N -> FC
```

其中前半部分主要负责从图像中提取局部特征，后半部分主要负责基于这些特征做分类或回归。

## 总体对比

| Component | 是否有参数 | 是否改变空间尺寸 | 核心计算行为 | 主要作用 |
| --- | --- | --- | --- | --- |
| Fully-Connected / Linear | 是 | 不涉及空间结构 | 矩阵乘法加偏置 | 融合全部输入特征，输出隐藏表示或类别分数 |
| Activation Function | 否 | 通常不改变形状 | 对每个元素独立做非线性变换 | 引入非线性，使网络能表达复杂函数 |
| Convolution Layer | 是 | 可能改变 `H, W` 和通道数 | 局部窗口与卷积核做点积 | 提取局部空间特征，共享参数 |
| Pooling Layer | 否 | 通常减小 `H, W` | 局部窗口取最大值或平均值 | 降采样，减少计算，增强局部平移鲁棒性 |
| Normalization | 通常有 | 通常不改变形状 | 标准化激活，再缩放平移 | 稳定训练，加快收敛，改善梯度传播 |

> 注：这里的“是否有参数”指是否有需要学习的参数。BatchNorm、LayerNorm 等归一化层通常有可学习的 `gamma` 和 `beta`。

## 1. Fully-Connected Layer

### 核心公式

```text
out = xW + b
```

如果输入 `x` 的形状是：

```text
x: N x D
W: D x M
b: M
out: N x M
```

其中：

- `N` 是 batch size。
- `D` 是每个样本的输入特征维度。
- `M` 是输出特征维度，分类任务中最后一层的 `M` 通常等于类别数。

### 计算行为

全连接层会让每个输出神经元都连接到输入中的所有特征。对第 `j` 个输出：

```text
out_j = x_1 * W_1j + x_2 * W_2j + ... + x_D * W_Dj + b_j
```

也就是说，它做的是一种全局特征加权求和。

### 作用

- 在 CNN 后半部分，把卷积层提取出的局部特征组合成更高层语义。
- 在分类网络最后一层，把隐藏表示转换成每个类别的 score。
- 能表达特征之间的全局组合关系，但参数量通常较大。

### 注意点

- 全连接层要求输入是向量，所以卷积特征通常要先 `Flatten`。
- 参数量是 `D * M + M`，当 `D` 很大时容易带来较高内存和过拟合风险。
- 最后一层 Linear 通常不直接接 ReLU，而是把 raw scores 交给 softmax loss 或 cross entropy loss。

## 2. Activation Function

### 核心思想

激活函数对输入逐元素进行非线性变换：

```text
out = f(x)
```

如果没有激活函数，多层线性层叠加后仍然等价于一个线性层：

```text
Linear(Linear(x)) = another Linear(x)
```

所以激活函数的核心作用是打破线性表达限制。

### 常见激活函数

#### ReLU

```text
f(x) = max(0, x)
```

特点：

- 正数原样通过，负数变成 0。
- 计算简单，梯度传播效果通常较好。
- 是 CNN 中最常见的默认激活函数。
- 缺点是负半轴梯度为 0，可能出现 dead ReLU。

#### Sigmoid

```text
f(x) = 1 / (1 + exp(-x))
```

特点：

- 输出范围是 `(0, 1)`。
- 适合表示概率或门控值。
- 当输入绝对值很大时梯度接近 0，容易梯度消失。
- 输出不是零中心，深层网络中训练可能较慢。

#### Tanh

```text
f(x) = tanh(x)
```

特点：

- 输出范围是 `(-1, 1)`。
- 相比 sigmoid 是零中心。
- 输入过大或过小时也会饱和，仍然可能梯度消失。

### 作用

- 引入非线性，使神经网络可以拟合复杂函数。
- 控制信息通过方式，例如 ReLU 会筛掉负值响应。
- 影响梯度传播，因此会直接影响训练速度和稳定性。

## 3. Convolution Layer

### 输入输出形状

对图像或 feature map，常见输入形状是：

```text
x: N x C_in x H x W
```

卷积核参数形状是：

```text
w: C_out x C_in x K_h x K_w
b: C_out
```

输出形状是：

```text
out: N x C_out x H_out x W_out
```

其中：

```text
H_out = 1 + (H + 2P - K_h) / S
W_out = 1 + (W + 2P - K_w) / S
```

`P` 是 padding，`S` 是 stride。实际使用时通常要求结果是整数。

### 计算行为

卷积层会让每个卷积核在输入的空间位置上滑动。每到一个位置，就取一个局部窗口，与卷积核做逐元素乘法再求和：

```text
out[n, f, i, j] =
    sum over c, p, q of x[n, c, i*S+p, j*S+q] * w[f, c, p, q] + b[f]
```

其中：

- `n` 表示第几个样本。
- `f` 表示第几个 filter，也就是输出通道。
- `i, j` 表示输出 feature map 的空间位置。
- 一个 filter 会生成一个输出通道。

### 作用

- 利用局部连接提取局部模式，例如边缘、角点、纹理、局部形状。
- 利用参数共享减少参数量：同一个 filter 在所有空间位置复用。
- 保留空间结构，让网络知道特征大概出现在图像的什么位置。
- 深层卷积会逐渐组合低级特征，形成更高级的语义特征。

### 超参数影响

#### Filter Size

- `K` 越大，每次看到的局部区域越大。
- 常见选择是 `3x3`、`5x5`、`7x7`。
- 多个小卷积堆叠可以获得更大的感受野，同时减少参数并增加非线性。

#### Stride

- `S = 1` 通常保持较密集的空间扫描。
- `S > 1` 会跳着扫描，使输出空间尺寸变小。
- stride 可以起到降采样作用。

#### Padding

- padding 在输入边界补 0。
- 可以控制输出尺寸。
- 常见设置是让卷积前后 `H, W` 不变，例如 `K=3, P=1, S=1`。

#### Number of Filters

- `C_out` 决定输出通道数。
- filter 越多，网络能学习的特征类型越多，但计算量和参数量也越大。

### 参数量

```text
params = C_out * C_in * K_h * K_w + C_out
```

例如：

```text
Conv(C_out=20, C_in=1, K=5)
params = 20 * 1 * 5 * 5 + 20 = 520
```

## 4. Pooling Layer

### 核心思想

池化层在每个通道内独立处理局部窗口，常见的是 Max Pooling：

```text
out[n, c, i, j] = max over local window of x[n, c, :, :]
```

如果是 `2x2` max pooling 且 stride 为 2，那么每个 `2x2` 小区域会被压缩成 1 个数。

### 输入输出形状

输入：

```text
x: N x C x H x W
```

输出：

```text
out: N x C x H_out x W_out
```

其中：

```text
H_out = 1 + (H - K_h) / S
W_out = 1 + (W - K_w) / S
```

池化层通常不改变通道数 `C`。

### 常见类型

#### Max Pooling

```text
out = max(window)
```

特点：

- 保留局部区域中响应最强的特征。
- 常用于图像 CNN。
- 直觉上表示“这个局部区域是否出现了某个特征”。

#### Average Pooling

```text
out = mean(window)
```

特点：

- 保留局部平均信息。
- 输出更平滑。
- 在一些网络末尾会使用 Global Average Pooling。

### 作用

- 减小空间尺寸，降低后续计算量。
- 扩大后续神经元的感受野。
- 提供一定局部平移鲁棒性：特征在小范围内移动，池化输出可能不变。
- 帮助控制过拟合，但会丢失一部分精确位置信息。

### 注意点

- 池化没有可学习参数。
- Max Pooling 反向传播时，梯度只传给前向传播中取得最大值的位置。
- 如果任务很依赖精确空间位置，例如分割或检测，需要谨慎使用过强的池化。

## 5. Normalization Layer

### 核心公式

归一化层通常先对激活值做标准化：

```text
x_hat = (x - mean) / sqrt(var + eps)
```

然后再用可学习参数做缩放和平移：

```text
out = gamma * x_hat + beta
```

其中 `eps` 用来防止除以 0。

### Batch Normalization

BatchNorm 在训练时通常对 mini-batch 维度统计均值和方差。

对全连接层：

```text
x: N x D
mean, var: 对 N 维度统计，每个特征维度各有一组统计量
```

对卷积层：

```text
x: N x C x H x W
mean, var: 通常对 N, H, W 统计，每个通道 C 各有一组统计量
```

训练时：

- 使用当前 mini-batch 的 mean 和 variance。
- 同时维护 running mean 和 running variance。

测试时：

- 使用训练期间累计的 running mean 和 running variance。

### Layer Normalization

LayerNorm 通常对单个样本内部的特征维度做归一化，不依赖 batch 中其他样本。

特点：

- 对 batch size 不敏感。
- 常用于 RNN、Transformer。
- 在 CNN 中 BatchNorm 更常见。

### 作用

- 稳定每层输入分布，让训练更平滑。
- 允许使用更大的 learning rate。
- 缓解梯度消失或梯度爆炸。
- 有轻微正则化效果，尤其是 BatchNorm 使用 mini-batch 统计时。

### 注意点

- BatchNorm 的行为在 train mode 和 eval mode 不同。
- batch size 太小时，BatchNorm 的统计量可能不稳定。
- `gamma` 和 `beta` 让归一化层不只是强行标准化，而是可以学习合适的尺度和偏移。

## 6. Flatten

### 核心行为

Flatten 会把多维 feature map 展平成向量：

```text
N x C x H x W -> N x (C * H * W)
```

例如：

```text
50 x 7 x 7 -> 2450
```

### 作用

- 连接卷积部分和全连接部分。
- 不改变数据数值，只改变张量形状。
- 展平后会丢失显式空间结构，因此通常放在卷积特征提取完成之后。

## 7. LeNet-5 风格结构示例

以输入 `1 x 28 x 28` 的灰度图为例：

| Layer | Output Size | Weight Size |
| --- | --- | --- |
| Input | `1 x 28 x 28` | - |
| Conv (`C_out=20, K=5, P=2, S=1`) | `20 x 28 x 28` | `20 x 1 x 5 x 5` |
| ReLU | `20 x 28 x 28` | - |
| MaxPool (`K=2, S=2`) | `20 x 14 x 14` | - |
| Conv (`C_out=50, K=5, P=2, S=1`) | `50 x 14 x 14` | `50 x 20 x 5 x 5` |
| ReLU | `50 x 14 x 14` | - |
| MaxPool (`K=2, S=2`) | `50 x 7 x 7` | - |
| Flatten | `2450` | - |
| Linear (`2450 -> 500`) | `500` | `2450 x 500` |
| ReLU | `500` | - |
| Linear (`500 -> num_classes`) | `num_classes` | `500 x num_classes` |

这个例子展示了 CNN 中常见的尺寸变化规律：

- 卷积层用 padding 保持空间尺寸，同时改变通道数。
- ReLU 不改变形状。
- Pooling 减小空间尺寸，不改变通道数。
- Flatten 把 `C x H x W` 变成一维向量。
- 全连接层把特征向量映射到隐藏维度或类别分数。

## 8. 从前向传播角度理解整个网络

一个图片分类 CNN 的前向传播可以理解成：

```text
image
-> convolution: 找局部模式
-> activation: 保留有用响应，引入非线性
-> pooling: 压缩空间尺寸，保留强响应
-> repeat: 组合成更高级特征
-> flatten: 转成向量
-> fully-connected: 全局整合特征
-> final score: 每个类别一个分数
```

更抽象地说：

- 卷积层回答：“局部有没有某种模式？”
- 激活函数回答：“这个响应是否应该通过？”
- 池化层回答：“这个区域里最强的响应是什么？”
- 归一化层回答：“这些激活值的尺度是否稳定？”
- 全连接层回答：“把所有特征组合起来后，应该输出什么？”

## 9. 易混点总结

### 卷积和全连接的区别

| 对比项 | Convolution | Fully-Connected |
| --- | --- | --- |
| 连接方式 | 局部连接 | 全连接 |
| 参数共享 | 是 | 否 |
| 保留空间结构 | 是 | 否 |
| 参数量 | 通常较少 | 通常较多 |
| 适合处理 | 图像、语音等有局部结构的数据 | 已经展平的全局特征 |

### 卷积和池化的区别

| 对比项 | Convolution | Pooling |
| --- | --- | --- |
| 是否有学习参数 | 有 | 没有 |
| 是否改变通道数 | 通常会 | 通常不会 |
| 主要计算 | 加权求和 | max 或 average |
| 主要作用 | 学习特征 | 降采样和压缩特征 |

### Activation 和 Normalization 的区别

| 对比项 | Activation | Normalization |
| --- | --- | --- |
| 主要目的 | 引入非线性 | 稳定数值分布 |
| 常见位置 | Conv/FC 后 | Conv/FC 后，Activation 前或后都可能 |
| 是否改变形状 | 不改变 | 不改变 |
| 是否有参数 | 通常没有 | 通常有 `gamma, beta` |

## 10. 复习口诀

```text
Conv 学局部特征，
ReLU 加非线性，
Pool 压缩空间尺寸，
Norm 稳定数值分布，
Flatten 接上分类头，
FC 融合全局信息。
```
