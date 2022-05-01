class RawSample:
    def __init__(self, request_ts=-1, start_ts=-1, end_ts=-1, node_num=-1, requested_sec=-1, queue_name=None):
        self.request_ts = request_ts
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.node_num = node_num
        self.requested_sec = requested_sec
        self.queue_name = queue_name

    def __str__(self):
        return self.__dict__.__str__()

class Preprocessor:
    # TODO
    def save(self, raw_sample_list, file_path):
        """
        保存预处理的结果。
        :param raw_sample_list: 待保存的RawSample数组
        :param file_path: 保存的位置
        """
        pass

    # TODO
    def load(self, file_path):
        """
        从保存的位置读取预处理过的数据。
        :param file_path: 保存的位置
        :return: 生成的RawSample数组
        """
        pass

    def preprocess(self, file_path):
        """
        原始文件转RawSample数组
        :param file_path: 原始文件位置
        :return: RawSample数组
        """
        raise NotImplementedError()