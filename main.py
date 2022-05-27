
from model.sklearn_linear_model import SklearnRegressionModel
from model.sklearn_logistic_regression import SklearnLogisticRegressionModel
from model.DecisionTreeClassifier import DecisionTreeClassifier
from model.DecisionTreeRegressor import DecisionTreeRegressor
from model.GradientBoostingClassifier import GB_Classifier
from model.GradientBoostingRegressor import GB_Regressor
from preprocess.preprocess_theta import PreprocessorTheta
from preprocess.Preprocessor_mira import PreprocessorMira
from preprocess.preprocess_taiyi import PreprocessorTaiyi
from sample.sample import sample_save, sample_load, to_sample_list
from sample.class_labeling import Labeler

if __name__ == '__main__':
    # preprocess
    preprocessor = PreprocessorTaiyi()
    # raw_list = preprocessor.preprocess('data/local/mira/sorted.csv')
    # preprocessor.save(raw_list,'data/local/mira/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/Taiyi/taiyi_raw_samples.txt')
    raw_list = preprocessor.load('data/local/mira/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/mira/RawSample_saved.txt')
    # generate sample
    # sample_list = to_sample_list(raw_list)
    # sample_save(sample_list, 'data/local/Taiyi/sample_saved.txt')
    sample_list = sample_load('data/local/mira/sample_saved.txt')

    # labelin
    AAE = list()
    PPE = list()
    labeler = Labeler(k=1)
    sample_list = labeler.label_samples(sample_list)

    # # run model
    model = GB_Regressor(sample_list, labeler, 7, 1, raw_list)

    # 修改聚类，取消聚类，并以queue_name作为分类标准。
    # model.label_queue_name()

    model.create_dataset()
    model.train()
    model.test()

    # # use model
    # print(model.predict(sample_list[0]))
