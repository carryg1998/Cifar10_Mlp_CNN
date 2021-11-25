import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable

import os
import numpy as np
from sklearn.metrics import classification_report

from utils.util import dim_redu, get_kfold_data, WriteExcelRow
from model.MLP import Mlp

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
        if (e + 1) % 5 == 0:
            #每训练5次迭代进行一次精确度评估
            model.eval()
            if cuda:
                X_test = X_test.cuda()
                y_test = y_test.cuda()
            y_pred = model(X_test)
            acc = np.mean((torch.argmax(y_pred, 1).cpu().numpy() == y_test.cpu().numpy()))
            print("Epoch", e+1, " acc:", acc)
        # torch.save(model.state_dict(), "logs/Epoch" + str(e+1) + ".pth")
        train_losses.append(train_loss / len(train_data))
    model.eval()
    if cuda:
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    y_pred = model(X_test)
    acc = np.mean((torch.argmax(y_pred, 1).cpu().numpy() == y_test.cpu().numpy()))

    return acc, model

if __name__ == "__main__":
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#
    #是否使用GPU训练
    cuda = True
    #分类数量
    class_num = 10
    # 训练迭代数
    epoch = 50
    #batch_size
    batch_size = 16
    #学习率
    learning_rate = 0.001
    # 降维比例和是否灰度处理,pca_ratio设置为1为不降维,灰度处理后特征数量为3072/3*pca_ratio
    pca_ratio = 1
    gray = False
    #隐藏层大小，设置为-1时为输入层大小除2
    hidden_c = -1
    #隐藏层数量，每一层隐藏层通道数固定
    hidden_num = 2
    #是否使用10折交叉验证,不使用时即计算F1值并将结果储存在表格中
    k_fold_10 = False
    #储存结果数据的表格路径
    path = "./result/result.xlsx"
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#

    data_path = "./cifar10/"

    #读取训练数据
    train_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    #读取测试数据
    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    #对图片降维
    train_X = dim_redu(train_data.data, gray, pca_ratio)
    test_X = dim_redu(test_data.data, gray, pca_ratio)

    #将数据由numpy格式转变为torch的张量模式
    X_train = torch.from_numpy(train_X).type(torch.FloatTensor)
    y_train = torch.tensor(train_data.targets, dtype=torch.long)
    X_test = torch.from_numpy(test_X).type(torch.FloatTensor)
    y_test = torch.tensor(test_data.targets, dtype=torch.long)

    #选择GPU训练
    device = torch.device("cuda:0")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if k_fold_10:
        acc_mean = 0
        for i in range(10):
            X_train, y_train, X_valid, y_valid = get_kfold_data(10, i, X_test, y_test)
            print("10折交叉验证，第", i+1, "次：")

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            input_size = X_test.shape[-1]

            #建立模型
            if hidden_c == -1:
                model = Mlp(input_size, input_size//2, class_num, hidden_num)
            else:
                model = Mlp(input_size, hidden_c, class_num, hidden_num)
            if cuda:
                model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)     #优化器

            acc, model = fit_one_train(epoch, train_loader, model, criterion, X_valid, y_valid, cuda)    #训练
            if gray:
                dim_reduce_ratio = 3072/3*pca_ratio
            else:
                dim_reduce_ratio = 3072*pca_ratio
            # torch.save(model.state_dict(), 'logs/dim%.2f-kfold%d-weight.pth'%(dim_reduce_ratio, i+1))
            print("第", i+1, "折模型正确率为", acc)
            acc_mean += acc
        print("十折交叉验证结束，平均正确率为:", acc_mean / 10)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        input_size = X_test.shape[-1]

        # 建立模型
        if hidden_c == -1:
            model = Mlp(input_size, input_size // 2, class_num, hidden_num)
        else:
            model = Mlp(input_size, hidden_c, class_num, hidden_num)
        if cuda:
            model.to(device)
        criterion = nn.CrossEntropyLoss()     #损失函数
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)     #优化器
        acc, model = fit_one_train(epoch, train_loader, model, criterion, X_test, y_test, cuda)    #训练
        if gray:
            dim_reduce_ratio = pca_ratio / 3
        else:
            dim_reduce_ratio = pca_ratio
        torch.save(model.state_dict(), 'logs/dim%.2f-weight.pth' % (dim_reduce_ratio))

        # 利用测试数据测试模型
        if cuda:
            X_test = X_test.cuda()
        y_pred = model(X_test)
        res = classification_report(torch.argmax(y_pred, 1).cpu(), y_test.cpu(), output_dict=True)
        print(classification_report(torch.argmax(y_pred, 1).cpu(), y_test.cpu()))
        f1_score = []
        for i in range(10):
            f1_score.append(res[str(i)]['f1-score'])
        print("每个类的F1值如下：", f1_score)
        f1_score_mean = res['macro avg']["f1-score"]

        #将降维后的特征数和最终平均f1指数写入表格
        WriteExcelRow([dim_reduce_ratio, f1_score_mean], path, -1, "features")
