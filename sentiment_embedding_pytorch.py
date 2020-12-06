# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 10:52 下午
# @File    : sentiment_embedding_pytorch.py

import torch
from torchtext import data
import random
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

"""
流程：
1.下载IMDB公开数据集,并处理
2.制作训练集和验证集（增加一个验证集）
3.读入glove词向量（这里会下载词向量）
4.建立模型
5.训练
6.测试
"""

# 1.下载IMDB公开数据集,并处理
SEED = 1234
torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子

# 在程序刚开始加这条语句可以提升一点训练速度,没什么额外开销。
torch.backends.cudnn.deterministic = True

# 用来定义字段的处理方法（文本字段,标签字段）
TEXT = data.Field(tokenize='spacy')  # Default: string.split
LABEL = data.LabelField(dtype=torch.float)  # Default: torch.long

# 这里会下载文件
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 2.制作训练集和验证集（增加一个验证集）
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# 3.读入glove词向量（这里会下载词向量）
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


# 4.建立模型
class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        # padding_idx:所有索引为padding_idx的都用0来填充。默认padding_idx=0,我们的词表{'<unk>': 0,'<pad>': 1},0和1是不参与计算的
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        # 使用lstm/gru也是可以的,模型写得简单了一些.
        embedded = self.embedding(text)  # [sent_len, batch_size, emb_dim]
        embedded = embedded.permute(1, 0, 2)  # [batch_size, sent_len, emb_dim]
        # 卷积核是(sent_len,1), 经过后-> [batch_size,1,emb_dim] 再squeeze-> [batch_size=64,emb_dim=100]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)  # [batch_size=64,output_dim=1]


INPUT_DIM = len(TEXT.vocab)  # 词个数=25002
EMBEDDING_DIM = 100  # 词嵌入维度
OUTPUT_DIM = 1  # 输出维度
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # PAD_IDX=1

model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


def count_parameters(model):
    """返回模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # UNK_IDX=0

# 把UNK_IDX=0和PAD_IDX=1的位置设置成tensor.zeros(这步骤应该不需要，只是为了看看效果)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()  # sigmoid+bec_loss
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    # 使用round去估计最近的,四舍五入.
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# 5.训练
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train()  # 训练模式，有时候训练时会用到dropout、归一化等方法,但是测试的时候不能用dropout等方法。

    for batch in iterator:
        optimizer.zero_grad()
        """
        batch有text和label两个属性:batch.text.shape=[sent_len,64], batch.label.shape=[64],  64=batch_size
        in : [sent_len,batch_size=64]
        out : [batch_size=64,output_dim=1]
        """
        # 压缩维度,不然跟batch.label维度对不上
        predictions = model(batch.text).squeeze(1)  # [batch_size=64]

        loss = criterion(predictions, batch.label)  # ([64],[64])计算损失
        acc = binary_accuracy(predictions, batch.label)  # ([64],[64])计算准确率

        loss.backward()
        optimizer.step()

        # loss.item()已经本身除以了len(batch.label)
        # 所以得再乘一次,得到一个batch的损失,累加得到所有样本损失。
        epoch_loss += loss.item() * len(batch.label)  # *64

        # （acc.item()：一个batch的正确率） *batch数 = 正确数
        # train_iterator所有batch的正确数累加。
        epoch_acc += acc.item() * len(batch.label)  # *64

        # 计算train_iterator所有样本的数量,应该是17500
        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len  # 损失,正确率


# 6.测试
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()  # 测试模式,不进行反向传播
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """一个epoch需要多长时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # 只要模型效果变好,就存模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
