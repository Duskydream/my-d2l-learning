import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数

batch_size = 256
num_epochs = 20

transform = transforms.Compose([transforms.ToTensor()]) #图像转为Tensor并归一化

#下载数据集，使用Fashion-MNIST
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

        # 第一层卷积  1,6,5x5,2
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2)

        #第一层池化
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #第二层卷积 6,16,5x5
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

        #第二层池化 2x2
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # 全连接层 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # Fashion-MNIST有10个类别

        self.relu = nn.ReLU() #使用ReLU激活

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5) # 展平，为全连接层做准备
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # 最后一层不需要 ReLU
        return x
    
net = SimpleCNN()

net = net.to(device)

#训练

#  定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) #使用adam优化器


for epoch in range(num_epochs):
    net.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in train_iter:
        #前向传播

        X, y = X.to(device), y.to(device)

        y_hat = net(X)
        loss = criterion(y_hat, y)
        
        #反向传播和优化
        optimizer.zero_grad() # 梯度清零
        loss.backward()       # 反向传播计算梯度
        optimizer.step()      # 更新参数
        
        #统计指标
        running_loss += loss.item()
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_iter):.4f}, Acc: {100 * correct / total:.2f}%')

# 简单评估
net.eval() # 设置为评估模式
correct = 0
total = 0
with torch.no_grad(): # 评估时不需要计算梯度
    for X, y in test_iter:

        X, y = X.to(device), y.to(device)

        outputs = net(X)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f'准确率: {100 * correct / total:.2f}%')