**Dynamic FL**



**Reference:**

1. Communication-Efﬁcient Learning of Deep Networks
   from  Decentralized Data
   1. Non-iid分法（dirichlet（利用先验和上一次的条件概率预测后验概率密度函数）/label）:
      1. Sort digit label, 60000个点划分成200*300, 然后给100个client每人2个(pathological non-iid partition)
      2. ![image-20220707031820058](/Users/qile/Library/Application Support/typora-user-images/image-20220707031820058.png)
      3. 在non-iid的情况下, 当C, B相同时, E的变化导致的performance变化并无规律
      4. **B = 10比B = 无穷效果好, especially in the non-IID case** (MNIST, 论文中提出)
      5. ![image-20220707113603625](/Users/qile/Library/Application Support/typora-user-images/image-20220707113603625.png)
      6. For very large numbers of local epochs, FedAvg can plateau or diverge.
      7. ![image-20220707113750509](/Users/qile/Library/Application Support/typora-user-images/image-20220707113750509.png)
2. Federated Learning on Non-IID Data_ A Survey
   1. Attribute skew
      1. non-overlapping skew
      2. partial overlapping skew
      3. full overlapping
   2. label skew
      1. label distribution skew
         1. label size imbalance (fixed number c label classes)
         2. label distribution imbalance (portion of label classes depends on Dir(B), Dir is the dirichlet distribution and B is the concentration parameter influencing the imbalance level, larger B -> more unbanlanced data partition)
      2. Label preference skew (暂时很少，没什么用)
         1. P(y|x) is different, x is data and y is label
   3. ![image-20220707131420179](/Users/qile/Library/Application Support/typora-user-images/image-20220707131420179.png)
   4. Data based approach
      1. 目标: modifying the distributions
      2. Data sharing
         1. 混合global shared dataset和local dataset做训练
         2. 需要data sharing
      3. Data augmentation
         1. replenishing the local imbalanced data with augmentations
         2. 比如根据label占比, 调整augmentation, 使用base data samples和decoded samples（只能被自己的数据解开的）构建一个新数据集
         3. 需要data sharing
   5. Algorithm based approach
      1. local fine-tuning
         1. ﬁnding a suitable initial shared model
            1. using meta learning
         2. combining local and global information
            1. regularization 加正则项
               1. minimize the disparity between the global and local models
            2. interpolation
               1. Global data + local data
               2. global model + local model
      2. Personalized layer
         1. 略
      3. Multi-task learning
      4. Knowledge distillation
         1. 做法: transfer information from large models to small ones
         2. federated transfer learning
         3. domain adaption
            1. eliminating the diﬀerences between data shards **between clients**
      5. lifelong learning
         1. 借鉴maintain the model accuracy without forgetting previously learnt tasks，task A学习的重要参数有penalty项，task B不能轻易修改
         2. 在fl中, 各个client标记重要参数，在重要参数上加惩罚项, 让global model更好
      6. structure adaption
         1. adaptive optimizers
         2. Group normalization
   6. System based approach
      1. Client clustering
         1. similarity of the loss value
            1. server拥有几个cluster model, 都传给client, client选择loss最小的那个更新
         2. other is the similarity of model weights
            1. 根据Model weights相似度聚类
      2. system level optimization
         1. 写包, 构建系统
3. Data-Free Knowledge Distillation for Heterogeneous Federated Learning
4. Mitigating Data Heterogeneity in Federated Learning with Data Augmentation
5. Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
   1. Non-iid 来源，使用了server momentum (Nesterov accelerated gradient)
6. XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning
   1. The core idea is to collect other devices’ encoded data samples that are decoded only using each device’s own data samples

7. Federated Learning with Non-IID Data
   1. non-iid 推导
   2. https://arxiv.org/abs/1806.00582




**Contributions:**

1. 提升训练效率
2. 解决non-iid问题



**Symbols:**

*E*(symbol check): gradient update的次数

N: sample size(row的数量)

B: batch size

M: 总client数量

C: active ratio=0.1

M_t: 每轮被选择的client数量 = M * C

T: Communication rounds

S: model size

P: Communication cost / round: 

		1. Upperbound: FedSgd: M_t * S * E
		1. Lowerbound: FedAvg: M_t * S * 1
		1. our method: M_t * S * X (1 <= X <= E)



**Assumptions:**

1. N一样
2. B一样

3. Global model update方式:
   1. average传到server的E的不同种类的client再返回



**Applications:**

1. Semi-supervised learning



**Experiments:**

1. Dataset:

   1. Mnist
   2. **Federated Extended MNIST**
   3. **Cifar-10**
   4. **Cifar-100**
   5. svhn
   6. Fashion-Mnist
   7. Sentiment140
   8. Shakespeare
   9. CelebA

2. 比较方法:

   1. FedAvg
   
   2. FedProx
      1. 对本地模型权重参数和全局模型权重参数的差异的限制
         1. add proximal term to local optimal function: u/2(w-w_t)^2 (w_t is the global model at round t， w is local model parameter, u is penalty factor)
         2. 问题:
            1. u选择
            2. nn.NLLLoss()
         3. 和fedavg的区别：
            1. change **local epoch E** to r_k^t - inexact minimizer (![image-20220807175349935](/Users/qile/Library/Application Support/typora-user-images/image-20220807175349935.png))
   
   3. FedEnsemble
   
      1. FedAvg + Ensemble
   
   4. FedGen
   
      1. 区别:
   
         1. client:
            1. 使用generator增强数据
            2. 改变loss func
   
         2. server:
            1. 构建generator
            2. 改变loss func
   
      2. softmax
   
         1. ![image-20220808174249934](/Users/qile/Library/Application Support/typora-user-images/image-20220808174249934.png)
   
      3. sigmoid
   
         1. ![img](https://pic4.zhimg.com/80/v2-7c4e9e545f0cc76bb5202fec4202f873_720w.jpg)
   
      4. softmax和sigmoid的区别？
   
      5. Cross Entropy Loss Function （假设是多项式分布）
   
         1. ![image-20220808172910954](/Users/qile/Library/Application Support/typora-user-images/image-20220808172910954.png)
   
      6. 为什么分类不使用mse loss （假设是高斯分布）:
   
         1. 主要原因是在分类问题中，使用sigmoid/softmx得到概率，配合MSE损失函数时，采用梯度下降法进行学习时，会出现模型一开始训练时，学习速率非常慢的情况
         2. mse不好收敛，有Local optimal, 不是strong convex
   
      7. 前置知识: knowledge distillation
   
         1. Knowledge distillation属于模型压缩的一种方法
   
         2. 为什么kd可以在减少模型参数的情况下获得和大模型差不多的效果？
   
            1. 模型的参数量和performance是非线性关系，存在边际效应，所以可以找一个optimal point
   
            2. KD的训练过程和传统的训练过程的对比
   
               1. 传统training过程(hard targets): 对ground truth求极大似然
               2. KD的training过程(soft targets): 用large model的class probabilities作为soft targets
   
            3. KD的训练过程为什么更有效?
   
               1. softmax层的输出，除了正例之外，负标签也带有大量的信息，比如某些负标签对应的概率远远大于其他负标签。而在传统的训练过程(hard target)中，所有负标签都被统一对待。也就是说，KD的训练方式使得每个样本给Net-S带来的信息量大于传统的训练方式。
               2. ![img](https://pic4.zhimg.com/80/v2-a9e90626c5ac6f64a7e04c89f6ce3013_720w.jpg)
               3. ![image-20220808183524104](/Users/qile/Library/Application Support/typora-user-images/image-20220808183524104.png)
               4. ![image-20220808190120115](/Users/qile/Library/Application Support/typora-user-images/image-20220808190120115.png)
   
               
   
   
   

**ToDos:**

1. Define data and problem in LaTex (problem formulation)
2. Define sudo-code in LaTex (done)
   1. **每一个命名都需要在input说明或者加一句说明语**
   2. **Symbol要清晰并且格式统一**
   3. **每个函数input, output要清楚（形参可变）**
   4. **每个变量要保证循环**
3. Proposal
   1. Abstract 
   2. Introduction(motivation)
      1. 利用communication budget
      2. 去除fedavg non-iid问题
   3. related work
   4. Proposed method
   5. timeline
4. 衔接local model at t and global model at t+1, 算出gradient后, 用optimizer更新。
5. quantity skew
6. **如何处理不同的算法？**
   1. **在train_classifier_fl中用if区分**
      1. fedavg
      2. fedprox
      3. fedensemble
      4. fedgen
      5. dynamicfl
      6. fedsgd
   2. **在server folder下建立不同的文件处理不同的算法**
   3. FedEnsemble到底是哪篇？
   4. 构建不同server算法的逻辑
7. 生成Non-iid数据
8. 跑通所有算法
9. 处理Federated Extended MNIST， Cifar-10， Cifar-100数据集
10. 如何验证算法work？
    1. 
11. Info
    1. bagging 求mean
    2. stacking 把多个Output当一个feature, 再过一遍模型
    3. Boosting 
    4. Bootstrap 有放回的均匀抽样
    5. Knowledge: soft label来训练更准确