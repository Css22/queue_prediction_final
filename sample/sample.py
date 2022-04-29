import heapq
class Sample:
    def __init__(self, cpu_hours = -1, cpus=-1, queue_load= -1, system_load=-1, actual_hour=-1, class_label=-1):
        self.cpu_hours = cpu_hours
        self.cpus = cpus
        self.queue_load = queue_load
        self.system_load = system_load
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
    sample_list = []
    request_ts_list = []

    for i in preprocessed_list:
        if len(request_ts_list) == 0 :
            tem = Sample(i.node_num * i.requested_hour,i.node_num,0,0,i.actual_hour)
            sample_list.append(tem)
        else:
            system_load = 0
            queue_load = 0

            while(len(request_ts_list) != 0):
                if request_ts_list[0].end_ts <= i.request_ts:
                    heapq.heappop(request_ts_list)
                else:
                    for job in request_ts_list:
                        system_load = system_load + job.node_num
                        if job.queue_name == i.queue_name:
                            queue_load = queue_load + job.node_num
                    break
            tem = Sample(i.node_num * i.requested_hour, i.node_num, queue_load,system_load , i.actual_hour)
            sample_list.append(tem)
        heapq.heappush(request_ts_list,i)
    return sample_list




