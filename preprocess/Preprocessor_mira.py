from preprocess.preprocess import Preprocessor, RawSample
from utils import time_utils

import pandas as pd
class PreprocessorMira(Preprocessor):

    def preprocess(self, file_path):
        dataset = pd.read_csv(file_path)
        dataset = dataset.sort_values(by='QUEUED_TIMESTAMP' , ascending=True)
        preprocessed_list = []
        for i in range(0,len(dataset['QUEUED_TIMESTAMP'])):
            tmp_raw = RawSample(time_utils.to_timestamp(dataset['QUEUED_TIMESTAMP'][i]),
                                time_utils.to_timestamp(dataset['START_TIMESTAMP'][i]),
                                time_utils.to_timestamp(dataset['END_TIMESTAMP'][i]),
                                dataset['NODES_REQUESTED'][i],
                                dataset['WALLTIME_SECONDS'][i],
                                dataset['QUEUE_NAME'][i],
                                dataset['ELIGIBLE_WAIT_SECONDS'][i])
            preprocessed_list.append(tmp_raw)
        return preprocessed_list