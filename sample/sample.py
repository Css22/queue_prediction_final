import heapq
import pickle
class Sample:
    def __init__(self, cpu_hours = -1, cpus=-1, queue_load= -1, system_load=-1, actual_sec=-1, class_label=-1, id=0):
        self.id = id
        self.cpu_hours = cpu_hours
        self.cpus = cpus
        self.queue_load = queue_load
        self.system_load = system_load
        self.actual_sec = actual_sec
        self.class_label = class_label

    def __str__(self):
        return self.__dict__.__str__()


# TODO
def sample_save(sample_list, file_path):
    """
    保存sample数组。
    :param raw_sample_list: 待保存的Sample数组
    :param file_path: 保存的位置
    """
    save_list = []
    for i in sample_list:
        tmp_list = [i.cpu_hours, i.cpus, i.queue_load, i.system_load, i.actual_sec, i.class_label]
        save_list.append(tmp_list)
    with open(file_path, 'wb') as text:
        pickle.dump(save_list, text)


# TODO
def sample_load(file_path):
    """
    导入sample数组。
    :param file_path: 保存的位置
    :return: Sample数组
    """
    with open(file_path, 'rb') as text:
        tmp_list = pickle.load(text)
    sample_list = [Sample(x[0], x[1], x[2], x[3], x[4], x[5], index) for index, x in enumerate(tmp_list)]
    return sample_list

# TODO
def to_sample_list(preprocessed_list):
    """
    将RawSample数组转Sample数组。不需要对class_label处理。
    :param preprocessed_list: RawSample数组
    :return: Sample数组
    """
    preprocessed_list.sort(key=lambda x: x.request_ts)
    sample_list = []
    request_ts_list = []

    for i in preprocessed_list:
        if len(request_ts_list) == 0 :
            tem = Sample(i.node_num * i.requested_sec, i.node_num, 0, 0, i.actual_sec, i.id)
            sample_list.append(tem)
        else:
            system_load = 0
            queue_load = 0

            while(len(request_ts_list) != 0):
                if request_ts_list[0].end_ts <= i.request_ts:
                    heapq.heappop(request_ts_list)
                else:
                    for job in request_ts_list:
                        if job.request_ts == i.request_ts:
                            continue
                        if job.start_ts < i.request_ts:
                            system_load = system_load + job.node_num
                        if job.start_ts >= i.request_ts:
                            queue_load = queue_load + 1
                    break
            tem = Sample(i.node_num * i.requested_sec, i.node_num, queue_load, system_load, i.actual_sec, i.id)
            sample_list.append(tem)
        heapq.heappush(request_ts_list,i)
    return sample_list




