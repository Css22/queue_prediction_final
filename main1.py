from model.MLP import RegressionModel
from model.sklearn_linear_model import SklearnRegressionModel
from model.sklearn_logistic_regression import SklearnLogisticRegressionModel
from preprocess.preprocess_theta import PreprocessorTheta
from preprocess.Preprocessor_mira import PreprocessorMira
from preprocess.preprocess_taiyi import PreprocessorTaiyi
from sample.sample import sample_save, sample_load, to_sample_list
from sample.class_labeling import Labeler

if __name__ == '__main__':
    print("mlp-theta-feature")
    # preprocess
    preprocessor = PreprocessorMira()
    # raw_list = preprocessor.preprocess('data/local/mira/sorted.csv')
    # preprocessor.save(raw_list,'data/local/mira/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/Taiyi/taiyi_raw_samples.txt')
    # raw_list = preprocessor.load('data/local/theta/RawSample_saved.txt')
    raw_list = preprocessor.load('data/local/mira/RawSample_saved.txt')
    # generate sample
    # sample_list = to_sample_list(raw_list)
    # sample_save(sample_list, 'data/local/theta/sample_saved.txt')
    sample_list = sample_load('data/local/mira/sample_saved.txt')

    # labelin
    AAE = list()
    PPE = list()
    labeler = Labeler(k=1)
    sample_list = labeler.label_samples(sample_list)

    # # run model
    # model = SklearnRegressionModel(sample_list, labeler, 7, 1, raw_list)
    # model = RegressionModel(sample_list, labeler, 4, 1, raw_list)
    model = SklearnLogisticRegressionModel(sample_list, labeler, 7, 1, raw_list)
    # 修改聚类，取消聚类，并以queue_name作为分类标准。
    # model.label_queue_name()
    model.create_dataset()
    model.train()
    model.test()
    # AAEtemp, PPEtemp = model.test()
    # AAE.append(AAEtemp)
    # PPE.append(PPEtemp)
    # print(AAE)
    # print(PPE)
    # model.save('linear_model.pkt')

    # # use model
    # print(model.predict(sample_list[0]))
