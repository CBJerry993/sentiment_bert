## 前言

传统embeding和bert在IMDB数据集进行情感分析的应用对比。

## 传统embeding

### 数据处理

- 来源：使用IMDB公开数据集。http://ai.stanford.edu/~amaas/data/sentiment/

- torchtext切分数据：使用可参考文档 https://pytorch.org/text/index.html
- 增加验证集：loss在测试集收敛了，那么在验证集的表现如何？也可判断测试集是否过拟合。
- 词向量：下载使用glove

 ```python
  TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
 ```

### 模型

具体参考代码，不需要太复杂的模型对于此场景足矣，比较简单。

### 训练

- 优化器：Adam（容易跳出局部最优），不精调（可使用随机梯度+动量）
- 损失函数：BCEWithLogitsLoss(适用于二分类问题=BCE+sigmoid)

### 测试

类似训练，略。

## Bert

### 模型

加载预训练模型

```python
bert_model = BertModel.from_pretrained('bert-base-uncased')
```

加载tokenizer

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

其他流程类似传统embeding，具体参考代码。

## 其他参考

- [知乎-torchtext的情感分析](https://zhuanlan.zhihu.com/p/94941514)



