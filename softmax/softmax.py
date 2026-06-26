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

train_losses = []
train_accs = []
test_accs = []

# ========== 初始化同步绘图 ==========
plt.ion()  # 开启交互模式
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()  # 创建双y轴，共享x轴

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
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
            
        epoch_loss += loss.item()
        num_batches += 1
        
    avg_loss = epoch_loss / num_batches
    train_acc = evaluate_accuracy(train_loader, W, b)
    test_acc = evaluate_accuracy(test_loader, W, b)
    
    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
