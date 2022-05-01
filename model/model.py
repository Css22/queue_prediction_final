class Model:
    def __init__(self, sample_list, labeler=None):
        self.sample_list = sample_list
        self.labeler = labeler
        self.train_dataset = list()
        self.test_dataset = list()
        self.create_dataset()

    def train(self):
        """
        主函数，在这里运行
        """
        pass

    def predict(self, sample):
        """
        预测一个sample的actual_hour
        :param sample: Sample
        :return: double, 输出的时间
        """
        pass

    def save(self, file_path):
        """
        保存model
        :param file_path:
        :return:
        """
        pass

    def load(self, file_path):
        """
        恢复model。不需要恢复sampler_list和labeler
        :param file_path:
        :return:
        """
        pass
    def create_dataset(self):
        """
        对于输入的sample_list进行训练集与测试集的划分。
        我们是采取训练训练集：测试集 = 8：2的比例进行划分。
        注意，这里进入的sample_list已经进行过打乱。
        """
        index = int(len(self.sample_list)/10 * 8)
        for i in range(0,index):
            self.train_dataset.append(self.sample_list[i])

        for i in range(index,len(self.sample_list)):
            self.test_dataset.append(self.sample_list[i])