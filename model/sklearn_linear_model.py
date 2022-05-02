import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch.nn import init

from model.model import Model


class SklearnRegressionModel(Model):
    def __init__(self, sample_list, labeler, n_feature, n_output):
        super().__init__(sample_list, labeler)
        self.n_feature = n_feature
        self.n_output = n_output
        self.model_list = []
        self.train_dataset = []
        self.test_dataset = []

        for i in range(0, self.labeler.k):
            train = list()
            test = list()
            self.train_dataset.append(train)
            self.test_dataset.append(test)
        self.create_dataset()

    """
       我们以数据集 训练，测试比例 8:2的比例进行来来行训练与测试，
       同时将训练好的模型按照聚类的类别保存到对应下标的model_list中
       """

    def train(self):
        train_list = []
        label_list = []
        for i in range(0, self.labeler.k):
            train = list()
            label = list()
            train_list.append(train)
            label_list.append(label)

        for i in range(0, self.labeler.k):
            for j in self.train_dataset[i]:
                tem_train_list = list()
                tem_train_list.append(math.log2(j.cpu_hours + 1))
                tem_train_list.append(j.cpus)
                tem_train_list.append(j.queue_load)
                tem_train_list.append(j.system_load)

                train_list[j.class_label].append(tem_train_list)
                label_list[j.class_label].append(math.log2(j.actual_sec + 1))

        # for i in range(0, len(train_list)):
        #     print(len(train_list[i]))

        for i in range(0, self.labeler.k):
            X_train, X_test, y_train, y_test = train_test_split(
                train_list[i], label_list[i],
                train_size=0.8, test_size=0.2, random_state=188
            )
            clf = LinearRegression(
            )

            # 使用训练数据来学习（拟合），不需要返回值，训练的结果都在对象内部变量中
            clf.fit(X_train, y_train)
            self.model_list.append(clf)

    # TODO
    def predict(self, sample):
        return_list = list()
        tmp_list = list()
        tmp_list.append(math.log2(sample.cpu_hours + 1))
        tmp_list.append(sample.cpus)
        tmp_list.append(sample.queue_load)
        tmp_list.append(sample.system_load)
        tmp_np = np.array(tmp_list)
        return_list.append(tmp_np)
        return_list = np.array(return_list)
        selected_model = self.model_list[sample.class_label]
        result = selected_model.predict(return_list)
        return result

    # TODO
    def save(self, file_path):
        pass

    # TODO
    def load(self, file_path):
        pass

    def create_dataset(self):
        """
        对于输入的sample_list进行训练集与测试集的划分。
        我们是采取训练训练集：测试集 = 8：2的比例进行划分。
        注意，这里进入的sample_list已经进行过打乱。
        """
        count = [0] * self.labeler.k
        for i in range(0, len(self.sample_list)):
            count[self.sample_list[i].class_label] = count[self.sample_list[i].class_label] + 1

        for i in range(0, len(self.sample_list)):

            index = count[self.sample_list[i].class_label] / 10 * 8

            if len(self.train_dataset[self.sample_list[i].class_label]) <= index:
                self.train_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

            else:
                self.test_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

    def test(self):
        test = []
        sums = [0 for _ in range(6)]
        nums = [0 for _ in range(6)]

        for i in range(0, len(self.test_dataset)):
            for j in range(0, len(self.test_dataset[i])):
                test.append(self.test_dataset[i][j])
        rate = 0

        for i in test:

            predict_time = self.predict(i)
            predict_time = 2 ** predict_time - 1
            actual_time = i.actual_sec

            print(str(actual_time)+" "+str(predict_time))
            rate = abs(predict_time - actual_time) / (actual_time + 0.1) + rate
            if actual_time <= 3600:  # 0-1
                sums[0] += abs(predict_time - actual_time)
                nums[0] += 1
            elif actual_time <= 3600 * 3:  # 1-3
                sums[1] += abs(predict_time - actual_time)
                nums[1] += 1
            elif actual_time <= 3600 * 6:  # 3-6
                sums[2] += abs(predict_time - actual_time)
                nums[2] += 1
            elif actual_time <= 3600 * 12:  # 6-12
                sums[3] += abs(predict_time - actual_time)
                nums[3] += 1
            elif actual_time <= 3600 * 24:  # 12-24
                sums[4] += abs(predict_time - actual_time)
                nums[4] += 1
            else:
                sums[5] += abs(predict_time - actual_time)
                nums[5] += 1

        print(sums)
        print(nums)
        avgs = [np.round(sums[i] / nums[i] / 3600, 2) for i in range(6)]
        print(avgs)
        AAE = np.round(sum(sums) / sum(nums) / 3600, 2)
        print('AAE :', end=' ')
        print(AAE)
        PPE = rate / sum(nums)
        print('PPE :', end=' ')
        print(PPE)
