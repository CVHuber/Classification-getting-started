import torchvision
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from resnet_model import resnet18, resnet34


def train(model, device, train_loader, optimizer):
    # 设置为训练模式，会更新BN和Dropout参数
    model.train()
    epoch_loss = 0
    # 训练
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将训练样本和掩模加载到GPU
        data, target = data.to(device), target.to(device)
        # 将数据喂入网络，获得一个预测结果
        output = model(data)
        # 通过交叉熵函数计算预测结果和掩模之间的loss
        loss = F.cross_entropy(output, target)
        # loss累加
        epoch_loss += loss.item()
        # 梯度清零
        optimizer.zero_grad()
        # 通过loss求出梯度
        loss.backward()
        # 使用Adam进行梯度回传
        optimizer.step()
    print('loss=%.3f' %  (100. * epoch_loss / len(train_loader.dataset)))

def test(model, device, test_loader):
    # 设置为测试模式，不更新BN和Dropout参数
    model.eval()
    correct = 0
    # 不更新梯度
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 从10个概率值中选出最大的
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    print('accuracy=%.3f' % (100. * correct/len(test_loader.dataset)))

# 程序入口
def main():
    # 指定GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 模型载入GPU
    model = resnet18().to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 设置CIFAR10训练集
    train_data = torchvision.datasets.CIFAR10('./dataset',
                        transform=transforms.Compose([transforms.ToTensor()]), download=True)
    # 设置CIFAR10测试集
    test_data = torchvision.datasets.CIFAR10('./dataset',
                        transform=transforms.Compose([transforms.ToTensor()]),  train=False, download=True)
    # 定义训练集的载入器
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    # 定义训练集的载入器
    test_loader = torch.utils.data.DataLoader(test_data)

    # 设置epoch次数
    for epoch in range(100):
        # 训练集函数
        train(model, device, train_loader, optimizer)
        # 测试机函数
        test(model, device, test_loader)
    


if __name__ == '__main__':
    main()