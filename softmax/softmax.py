from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader 
import torch.nn.functional as F

trans = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root='.',      # 保存位置
    train=True,         # 训练集
    transform=trans,
    download=True       # 自动下载
)

test_dataset = datasets.FashionMNIST(
    root='.',  # 保存位置
    train=False,        # 测试集
    transform=trans,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print(len(train_dataset))
print(len(test_dataset))

img, label = train_dataset[0]

# Dataloader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,      # 每批次64张图片
    shuffle=True        # 打乱数据
)

X, y = next(iter(train_loader))

print(X.shape)
print(y.shape)  

# flatten

X =  X.reshape(X.shape[0], -1)

# Softmax

W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)


def evaluate_accuracy(data_loader,W,b):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.reshape(X.shape[0], -1)
            logits = X @ W + b
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
    return correct / total

lr = 0.1
num_epochs = 5

for epoch in range(num_epochs):
    for X, y in train_loader:
        X = X.reshape(X.shape[0], -1)
        
        # forward
        logits = X @ W + b
        loss = F.cross_entropy(logits, y)
        
        # backward
        loss.backward()
        
        # update
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()
    
    train_acc = evaluate_accuracy(train_loader, W, b)
    test_acc = evaluate_accuracy(test_loader, W, b)
    print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


print (f"Final Train Acc: {evaluate_accuracy(train_loader, W, b):.4f}")