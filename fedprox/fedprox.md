# Federated Optimization in Heterogeneous Networks导读

[TOC]

## Source

论文来源: 《[Federated Optimization in Heterogeneous Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.06127)》

这篇文章基于FedAvg的基础上，针对Non-IID数据做出了一些轻微的调整。

传统的FedAvg:

在每个 Communication Round 内，参与更新的 **K** 个设备在本地 SGD 迭代 **E** epochs，然后在将模型上传到 Server 端进行聚合。一方面，本地迭代次数 **E** 的增大能减少通信成本；另一方面，不同 local objectives ，比如梯度下降在本地迭代次数过多后容易偏离全局最优解，影响收敛。

文章指出**直接 drop 掉这些用户或者单纯把他们未迭代完成的模型进行聚合都会严重影响收敛的表现**。因为丢掉的这些设备可能导致模型产生 bias，并且减少了设备数量也会对结果精度造成影响。



## Proximal term

根据FedAvg的经验，优化目标是最小化经验损失Epirical risk:
$$
{min_w f(w)} = \sum_{k=1}^{N} {p_k}{F_k(w)} = \mathbb{E}[F_k(w)]
$$
其中
$$
p_k \geqslant 0 , \sum_{k=1}^{K}{p_k} = 1,p_k = \frac{n_k}{n}
$$
作者引入了额外的一个叫做proximal term的东西。
$$
\underset {\omega}{min}h(w;w^t)_k = F_k(w) + \frac{\mu}{2}\mid\mid w - w^t \mid\mid ^ 2
$$

## code

本地更新伪代码

### Proximal term

```python
def train():
    for epoch in range(Epochs):
        seq = seq.to(Device)
        label = label.to(Device)
        y_predict = model(seq)
        optimizer.zero_grad()
        
        for w,w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)#L2范数
        
        loss = loss_function(y_pred,label) + (mu / 2) * proximal_term
        loss_backward()
        optimizer.step()
```

在原有的loss上加了一个近端项：

```python
for w, w_t in zip(model.parameters(), global_model.parameters()):
    proximal_term += (w - w_t).norm(2)
```

### gamma-inexact solution

For a function 
$$
h(w;w_0) = F(w) + \frac{mu}{2}\mid\mid w - w_0 \mid\mid ^ 2
$$
and
$$
\gamma \in [0,1]
$$
we say w* is a y-inexact solution of h(w;w0)

conclusion:

总之只要找到一个w*  使h(w;w0)最小，且满足
$$
\mid\mid \nabla h(w^*;w_0) \mid \mid \leqslant \gamma \mid\mid \nabla h(w_0;w_0) \mid\mid
$$
这里w*是h的一个gamma不精确的解。



## Q&A

- **When would  drop these clients?**
- **Why many epochs will lead to drop clients?**
- **it seems that selected gamma is smaller and the solution of loss function will be more precise? isn't it?**

​		gamma selected is about converage problem, Involving complex mathematical proofs

- **how to select a good gamma to fit model? author didn't show in this paper, hearing that he will supply it.**

​		
