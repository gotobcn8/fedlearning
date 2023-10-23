# Communication-Efficient Learning of Deep Networks from Decentralized Data导读

[TOC]

## abstract

> Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos. However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches. We advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning. We present a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10–100× as compared to synchronized stochastic gradient descent.

it tells the development of federated learning tendency. nothing useful

## introduction

>  phones and tablets are the primary computing devices for many people 
>
> Federated Learning Ideal problems for federated learning have the following properties: 1) Training on real-world data from mobile devices provides a distinct advantage over training on proxy data that is generally available in the data center. 2) This data is privacy sensitive or large in size (compared to the size of the model), so it is preferable not to log it to the data center purely for the purpose of model training (in service of the focused collection principle). 3) For supervised tasks, labels on the data can be inferred naturally from user interaction.
>
> Federated Optimization We refer to the optimization problem implicit in federated learning as federated optimization, drawing a connection (and contrast) to distributed optimization. Federated optimization has several key properties that differentiate it from a typical distributed optimization problem: 
>
> **• Non-IID** The training data on a given client is typically based on the usage of the mobile device by a particular user, and hence any particular user’s local dataset will not be representative of the population distribution. 
>
> **• Unbalanced** Similarly, some users will make much heavier use of the service or app than others, leading to varying amounts of local training data.
>
> **• Massively distributed** We expect the number of clients participating in an optimization to be much larger than the average number of examples per client.
>
>  **• Limited communication** Mobile devices are frequently offline or on slow or expensive connections.

阐述了目前需要优化的四大方向：

- 非独立同分布数据
- 用户数据的不平衡
- 客户端分布
- 通信次数的限制

Non-IID is Non-Independent Identical Distributed

divided to three cases:

- Non Independent but Identical Distributed
- Independent but Not Identical Distributed
- Not Independent and Not Identical Distributed

> In this work, our emphasis is on the non-IID and unbalanced properties of the optimization

本次任务主要解决non-IID和数据不平衡的优化

> These issues are beyond the scope of the current work; instead, we use a controlled environment that is suitable for experiments, but still addresses the key issues of client availability and unbalanced and non-IID data. We assume a synchronous update scheme that proceeds in rounds of communication. There is a fixed **set of K clients**, each with a fixed local dataset. At the beginning of each round, a random **fraction** **C of clients is selected**, and the server sends the current global algorithm state to each of these clients (e.g., the current model parameters). We only select a fraction of clients for efficiency, as our experiments show diminishing returns for adding more clients beyond a certain point. Each selected client then performs local computation based on the global state and its local dataset, and sends an update to the server. The server then applies these updates to its global state, and the process repeats. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas ¨ While we focus on non-convex neural network objectives, the algorithm we consider is applicable to any finite-sum objective of the form

$$
\underset {\omega \in \mathbb{R}^d}{minf(w)}
$$

 where 
$$
f(w) \overset{def}{=} \frac{1}{n}\sum_{i=1}^n f_i(w)
$$


f(w) 是loss  function，fi(w)是第i个客户端的loss，

> For a machine learning problem, we typically take fi(w) = `(xi , yi ; w), that is, the loss of the prediction on example (xi , yi) made with model parameters w. We assume there are K clients over which the data is partitioned, with Pk the set of indexes of data points on client k, with nk = |Pk|. Thus, we can re-write the objective (1) as

K个客户端的数据被区分开，Pk表示第k个客户端的索引集合，nk = L1范数（pk）

$$
f(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w) where F_k(w) = \frac{1}{n_k}\sum_{i \in \mathcal{p_k}}f_i(w)
$$


全局的loss等于每个客户端loss的加权loss



## FedSGD

> SGD can be applied naively to the federated optimization problem, where a single batch gradient calculation (say on a randomly selected client) is done per round of communication. This approach is computationally efficient, but requires very large numbers of rounds of training to produce good models (e.g., even using an advanced approach like batch normalization, Ioffe and Szegedy [21] trained MNIST for 50000 steps on minibatches of size 60). We consider this baseline in our CIFAR-10 experiments.
>
> In the federated setting, there is little cost in wall-clock time to involving more clients, and so for our baseline we use large-batch synchronous SGD; experiments by Chen et al. [8] show this approach is state-of-the-art in the data center setting, where it outperforms asynchronous approaches. To apply this approach in the federated setting, we select a Cfraction of clients on each round, and compute the gradient of the loss over all the data held by these clients. Thus, C controls the global batch size, with C = 1 corresponding to full-batch (non-stochastic) gradient descent.2 We refer to this baseline algorithm as FederatedSGD (or FedSGD)

**FedSGD**是最初对联邦学习构想的实现，每轮选择其中一个客户端C，选择C内的全批次数据进行迭代。

这样的方法对比传统的机器学习是十分有效的，但是对于联邦学习来说它需要庞大的训练轮数来拟合一个良好的模型。

## FedAvg

FedAvg，选择C个客户端，E为训练的轮次，B为训练的批次（无穷代表全批次），同理

FedAvg(C=1, E = 1, B = infinite) = FedSGD

|      | FedAvg        | FedSGD  |
| ---- | ------------- | ------- |
|      | C = faction c | C = 1   |
|      | E = epochs    | E = 1   |
|      | B = Batch     | B = all |



## Algorithm

Algorithm 1 FederatedAveraging. TheK clients are
indexed by k; B is the local minibatch size, E is the number
of local epochs, and learning_rate is the learning rate.

```python
def server():    
    initialize w[T][K],wfinal[T]
    for each round t = 1,2,...,do
        m = max(C * K,1)
        St = random set(m)
        for each client k in St parallel do
            w[t+1][k] = ClientUpdate(k,w[t][k])
        wfinal[t+1] = WeightSum(w[t+1],n[K])


def ClientUpdate(k,w):
    initialize batches
    for e in epoches:
        for b in batches:
            w = w - learning_rate * derative(loss(w))
    return w

def WeightSum(w[T],n[K]):
    ws = 0
    for k in K:
        ws += (n[k] / n_total) * w[k]
    return ws
```

## Question

### parameter consistency

- now all the parameter seems to be consistent, but when facing Non-IID data, the model may not fit well when the data is drifting, so can different client fitting schemes be specified for different data drifting?


​		往往真实情况都是数据异构性的，参数一致性是否有改进的空间？

