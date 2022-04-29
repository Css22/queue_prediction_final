class Sample:
    def __init__(self, requested_hour = -1, requested_node=-1, queue_load= -1, node_load=-1, actual_hour=-1, class_label=-1):
        self.requested_hour = requested_hour
        self.requested_node = requested_node
        self.queue_load = queue_load
        self.node_load = node_load
        self.actual_hour = actual_hour
        self.class_label = class_label


# TODO
def sample_save(sample_list, file_path):
    """
    保存sample数组。
    :param raw_sample_list: 待保存的Sample数组
    :param file_path: 保存的位置
    """
    raise NotImplementedError()


# TODO
def sample_load(file_path):
    """
    导入sample数组。
    :param file_path: 保存的位置
    :return: Sample数组
    """
    raise NotImplementedError()

# TODO
def to_sample_list(preprocessed_list):
    """
    将RawSample数组转Sample数组。不需要对class_label处理。
    :param preprocessed_list: RawSample数组
    :return: Sample数组
    """
    raise NotImplementedError()
