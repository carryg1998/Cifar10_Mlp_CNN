import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable

import os
import numpy as np
from sklearn.metrics import classification_report

from model.CNN import CNNnet

def fit_one_train(epoch, train_loader, model, criterion, X_test, y_test, cuda):
    train_losses = []
    print("Start training")
    for e in range(epoch):

        train_loss = 0
        model.train()
        for X, y in train_loader:
            if cuda:
                X = X.cuda()
                y = y.cuda()
            X_data = Variable(X)
            out = model(X_data)

            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print("Epoch", e+1, " loss:", train_loss / len(train_data))

        if (e+1) == epoch:
            model.eval()
            if cuda:
                X_test = X_test.cuda()
            y_pred = model(X_test)
            acc = np.mean((torch.argmax(y_pred, 1).cpu().numpy() == y_test.cpu().numpy()))
            print("Epoch", e + 1, " acc:", acc)
            # torch.save(model.state_dict(), "logs/Epoch" + str(e+1) + ".pth")
            res = classification_report(torch.argmax(y_pred, 1).cpu(), y_test.cpu(), output_dict=True)
            f1_score = []
            for i in range(10):
                f1_score.append(res[str(i)]['f1-score'])
            print("每个类的F1值如下：", f1_score)
            f1_score_mean = res['macro avg']["f1-score"]
            print("F1平均值：", f1_score_mean)
            print(classification_report(torch.argmax(y_pred, 1).cpu(), y_test.cpu()))

        train_losses.append(train_loss / len(train_data))

    return acc, f1_score, f1_score_mean, model

if __name__ == "__main__":
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#
    #是否使用GPU训练
    cuda = True
    #分类数量
    class_num = 10
    # 训练迭代数
    epoch = 50
    # batch_size
    batch_size = 16
    #学习率
    learning_rate = 0.001
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#

    data_path = "./cifar10/"

    train_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    X_train = torch.from_numpy(train_data.data).type(torch.FloatTensor).permute(0, 3, 1, 2).contiguous()
    y_train = torch.tensor(train_data.targets, dtype=torch.long)
    X_test = torch.from_numpy(test_data.data).type(torch.FloatTensor).permute(0, 3, 1, 2).contiguous()
    y_test = torch.tensor(test_data.targets, dtype=torch.long)

    device = torch.device("cuda:0")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    input_size = X_test.shape[-1]

    model = CNNnet(class_num)
    if cuda:
        model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    acc, f1_score, f1_score_mean, model = fit_one_train(epoch, train_loader, model, criterion, X_test, y_test, cuda)
    print("精度为：", acc)
    print("每个类的F1值如下：", f1_score)
    print("平均F1值为：", f1_score_mean)
