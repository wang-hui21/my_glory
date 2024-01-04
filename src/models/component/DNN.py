# -*- coding: utf-8 -*-
# @Time : 2024/1/4 21:43
# @Author : Wang Hui
# @File : DNN
# @Project : my_glory
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 user_embeddings 和 news_embeddings 是两个张量
user_embeddings = torch.randn(1, 100)  # 假设用户嵌入的维度为 100
news_embeddings = torch.randn(1, 100)  # 假设新闻嵌入的维度为 100


# 构建神经网络模型
class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 创建模型实例
model = DNN(input_size=100)  # 假设输入的维度为 100

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将 user_embeddings 和 news_embeddings 转为浮点数张量
user_embeddings = user_embeddings.float()
news_embeddings = news_embeddings.float()

# 模型训练
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    output = model(user_embeddings)

    # 计算损失
    loss = criterion(output, news_embeddings)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型推理
with torch.no_grad():
    model.eval()
    predicted_prob = model(user_embeddings)
    print("Predicted Probability:", predicted_prob.item())
