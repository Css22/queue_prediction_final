import numpy as np
import torch

from model.model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import  init

class LinearNet(nn.Module) :
    def __init__(self, n_feature, n_output):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(n_feature, 20)
        self.linear2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, n_output)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        y = self.output(y)
        return y

class RegressionModel(Model):
    def __init__(self, sample_list, labeler, n_feature, n_output, model_list = list()):
        super().__init__(sample_list, labeler)
        self.n_feature = n_feature
        self.n_output = n_output

        if len(model_list) == 0:
            self.model_list = [None] * labeler.k
        else:
            self.model_list = model_list

    """
       我们以数据集 训练，测试比例 8:2的比例进行来来行训练与测试，
       同时将训练好的模型按照聚类的类别保存到对应下标的model_list中
       """
    def train(self):
        train_list = [list()] * self.labeler.k
        label_list = [list()] * self.labeler.k

        for i in self.train_dataset:
            tem_train_list = list()
            tem_train_list.append(i.cpu_hours)
            tem_train_list.append(i.cpus)
            tem_train_list.append(i.queue_load)
            tem_train_list.append(i.system_load)

            train_list[i.class_label].append(tem_train_list)
            label_list[i.class_label].append(i.actual_hour)


        for i in range(0,self.labeler.k):

            n_epoch = 20
            batch_size = 32

            features = torch.from_numpy(np.array(train_list[i]))
            labels = torch.from_numpy(np.array(label_list[i]))
            labels = torch.tensor(labels, dtype=torch.double)

            dataset = Data.TensorDataset(features, labels)
            data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

            net = LinearNet(self.n_feature,self.n_output).to(device='cuda')
            net = net.double()


            init.normal_(net.linear1.weight, mean=0, std=0.01)
            init.constant_(net.linear1.bias, val=0)
            init.normal_(net.linear2.weight, mean=0, std=0.01)
            init.constant_(net.linear2.bias, val=0)
            init.normal_(net.output.weight, mean=0, std=0.01)
            init.constant_(net.output.bias, val=0)

            loss = nn.MSELoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
            loss = loss.cuda()

            for epoch in range(1, n_epoch + 1):
                for X, y in data_iter:
                    X, y = X.cuda(), y.cuda()
                    output = net(X)
                    l = loss(output, y.squeeze())
                    optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
                    l.backward()
                    optimizer.step()
                print('epoch %d, loss: %f' % (epoch, l.item()), end=' ------')
                print()


    # TODO
    def predict(self, sample):
        pass

    # TODO
    def save(self, file_path):
        pass

    # TODO
    def load(self, file_path):
        pass