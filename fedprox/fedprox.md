# Federated Optimization in Heterogeneous Networks导读

## Source

论文来源: 《[Federated Optimization in Heterogeneous Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.06127)》

这篇文章基于FedAvg的基础上，针对Non-IID数据做出了一些轻微的调整。

传统的FedAvg:

在每个 Communication Round 内，参与更新的 **K** 个设备在本地 SGD 迭代 **E** epochs，然后在将模型上传到 Server 端进行聚合。一方面，本地迭代次数 **E** 的增大能减少通信成本；另一方面，不同 local objectives ，比如梯度下降在本地迭代次数过多后容易偏离全局最优解，影响收敛。

文章指出**直接 drop 掉这些用户或者单纯把他们未迭代完成的模型进行聚合都会严重影响收敛的表现**。因为丢掉的这些设备可能导致模型产生 bias，并且减少了设备数量也会对结果精度造成影响。

question:

When would exist drop these clients?

Why epochs is too much will drop clients?



## Proximal term

根据FedAvg的经验，优化目标是最小化经验损失Epirical risk:
$$
min_wf(w) = \sum_{k=1}^{N} p_kF_k(w) = \mathbb{E}[F_k(w)]
$$
其中
$$
p_k \geqslant 0 , \sum_{k=1}^{K}{p_k} = 1,p_k = \frac{n_k}{n}
$$
作者引入了额外的一个叫做proximal term的东西。
$$
\underset {\omega}{min}h(w;w^t)_k = F_k(w) + \frac{\mu}{2}\mid\mid w - w^t \mid\mid ^ 2
$$
