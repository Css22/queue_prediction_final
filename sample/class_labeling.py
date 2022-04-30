from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np


class Labeler:
    def __init__(self, center=np.array([])):
        self.center = center

    # TODO

    def label_samples(self, sample_list):
        """
            给samples聚类，给Sample.class_label打标签
            :param sample_list: Sample数组
            :return: 打好标签的Sample数组
            """
        k = 4;
        processingList = list()
        # 读取sample的各个特征,并将这些特征放入processingList中处理
        for sample in sample_list:
            array = list()
            array.append(sample.cpu_hours)
            array.append(sample.cpus)
            array.append(sample.queue_load)
            array.append(sample.system_load)
            processingList.append(array)
        processingNp = np.array(processingList)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(processingNp)
        self.center = kmeans.cluster_centers_;
        y_kmeans = kmeans.predict(processingNp)

        num_list = []
        for i in range(0,k):
            num_list.append(0)

        for i in y_kmeans:
            num_list[i] = num_list[i] +1

        for i in range(0,k):
            print('cluster', end=' ')
            print(i,': ',num_list[i])

        for i in range(0, len(sample_list)):
            sample_list[i].class_label = y_kmeans[i]
            # print(sample_list[i].class_label)
        distortions = []
        for i in range(1, k + 1):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=3000,
                tol=1e-04, random_state=0
            )
            km.fit(processingNp)

            # inertia: Sum of squared distances of samples to their closest cluster center,
            # weighted by the sample weights if provided.
            distortions.append(km.inertia_)

        # plot
        plt.plot(range(1, k + 1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()

    # TODO
    def label(self, sample):
        """
        给Sample.class_label打标签
        :param sample: Sample
        :return: int, 标签
        """
        array = list()
        array.append(sample.cpu_hours)
        array.append(sample.cpus)
        array.append(sample.queue_load)
        array.append(sample.system_load)
        vector = np.array(array)

        maxNumber = float("inf")
        for i in range(0, len(self.center)):
            temp = np.linalg.norm(self.center[i] - vector)
            if temp <= maxNumber:
                sample.class_label = i
                maxNumber = temp

    # TODO
    def load(self, file_path):
        """
        导入labeler
        :param file_path:
        """
        pass

    # TODO
    def save(self, file_path):
        """
        导出labeler
        :param file_path:
        """
        pass
