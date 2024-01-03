# -*- coding: utf-8 -*-
# @Time : 2024/1/2 11:29
# @Author : Wang Hui
# @File : new_bert
# @Project : my_glory
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class Bert_news(nn.Module):
    def __init__(self):
        super(Bert_news, self).__init__()
        self.bert = BertModel.from_pretrained('bert')
        self.linear = nn.Linear(768, 400)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.linear(pooler_output)
        return logits

# 输入文本
text = ["Your input text here.", "Please stand up"]
labels = torch.tensor([1, 0])  # 你需要提供相应的标签

# 对文本进行编码和处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(text, return_tensors='pt', padding=True)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 初始化模型
num_classes = 2  # 你的分类任务的类别数
model = BertClassifier(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 查看训练后的模型输出
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs, dim=1)
    print(predictions)
