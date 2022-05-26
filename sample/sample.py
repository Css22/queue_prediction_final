import heapq
import pickle


class Sample:
    def __init__(self, cpu_hours=-1, cpus=-1, queue_load=-1, system_load=-1, actual_sec=-1, class_label=-1, id=0,
                 future_node_load=0, future_requested_sec_load=0,
                 future_load=0, queue_name=None):
        self.id = id
        self.cpu_hours = cpu_hours
        self.cpus = cpus
        self.queue_load = queue_load
        self.system_load = system_load
        self.actual_sec = actual_sec
        self.future_load = future_load
        self.future_node_load = future_node_load
        self.future_requested_sec_load = future_requested_sec_load
        self.class_label = class_label
        self.queue_name = queue_name

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
        tmp_list = [i.cpu_hours, i.cpus, i.queue_load, i.system_load, i.actual_sec, i.future_load,
                    i.future_node_load, i.future_requested_sec_load, i.queue_name, i.id]
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
    sample_list = [Sample(cpu_hours=x[0], cpus=x[1], queue_load=x[2], system_load=x[3], actual_sec=x[4], future_load=x[5],
                          future_node_load=x[6], future_requested_sec_load=x[7], queue_name=x[8], id=x[9]) for index, x in enumerate(tmp_list)]
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
        if preprocessed_list.index(i) <= 500:
            heapq.heappush(request_ts_list, i)
            continue
        if preprocessed_list.index(i) >= (len(preprocessed_list) - 500):
            break
        if len(request_ts_list) == 0:
            future_load = 0
            future_node_load = 0
            future_requested_sec_load = 0
            index = preprocessed_list.index(i)
            for t in range(index + 1, len(preprocessed_list)):
                if preprocessed_list[t].request_ts > i.start_ts:
                    break
                if preprocessed_list[t].node_num * preprocessed_list[
                    t].requested_sec < i.node_num * i.requested_sec:
                    future_load = future_load + 1

                if preprocessed_list[t].node_num < i.node_num:
                    future_node_load = future_node_load + 1

                if preprocessed_list[t].requested_sec < i.requested_sec:
                    future_requested_sec_load = future_requested_sec_load + 1

            tem = Sample(cpu_hours=i.node_num * i.requested_sec, cpus=i.node_num, queue_load=0,
                         system_load=0, actual_sec=i.actual_sec, id=i.id,
                         future_node_load=future_node_load, future_requested_sec_load=future_requested_sec_load,
                         future_load=future_load, queue_name=i.queue_name)
            sample_list.append(tem)
        else:
            system_load = 0
            queue_load = 0
            future_load = 0
            future_node_load = 0
            future_requested_sec_load = 0
            while len(request_ts_list) != 0:
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

                    index = preprocessed_list.index(i)
                    for t in range(index + 1, len(preprocessed_list)):
                        if preprocessed_list[t].request_ts > i.start_ts:
                            break
                        if preprocessed_list[t].node_num * preprocessed_list[
                            t].requested_sec < i.node_num * i.requested_sec:
                            future_load = future_load + 1

                        if preprocessed_list[t].node_num < i.node_num:
                            future_node_load = future_node_load + 1

                        if preprocessed_list[t].requested_sec < i.requested_sec:
                            future_requested_sec_load = future_requested_sec_load + 1

                    break

            tem = Sample(cpu_hours=i.node_num * i.requested_sec, cpus=i.node_num, queue_load=queue_load,
                         system_load=system_load, actual_sec=i.actual_sec, id=i.id,
                         future_node_load=future_node_load, future_requested_sec_load=future_requested_sec_load,
                         future_load=future_load, queue_name=i.queue_name)
            sample_list.append(tem)
        if preprocessed_list.index(i) % 1000 == 0:
            print(preprocessed_list.index(i),len(preprocessed_list))
        heapq.heappush(request_ts_list, i)
    return sample_list
