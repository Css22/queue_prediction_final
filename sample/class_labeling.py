class Labeler:
    def __init__(self):
        pass

    # TODO
    def label_samples(self, sample_list):
        """
        给samples聚类，给Sample.class_label打标签
        :param sample_list: Sample数组
        :return: 打好标签的Sample数组
        """
        raise NotImplementedError()

    # TODO
    def label(self, sample):
        """
        给Sample.class_label打标签
        :param sample: Sample
        :return: int, 标签
        """
        pass

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
