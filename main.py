from model.linear_regression import RegressionModel
from model.sklearn_linear_model import SklearnRegressionModel
from preprocess.preprocess_theta import PreprocessorTheta
from preprocess.Preprocessor_mira import PreprocessorMira
from preprocess.preprocess_taiyi import PreprocessorTaiyi
from sample.sample import sample_save, sample_load, to_sample_list
from sample.class_labeling import Labeler

if __name__ == '__main__':
    # preprocess
    preprocessor = PreprocessorMira()
    # # raw_list = preprocessor.preprocess('data/local/Taiyi/sorted.csv')
    # preprocessor.save(raw_list,'data/local/mira/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/Taiyi/taiyi_raw_samples.txt')
    raw_list = preprocessor.load('data/local/theta/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/mira/RawSample_saved.txt')

    # generate sample
    sample_list = to_sample_list(raw_list)
    # sample_save(sample_list, 'data/local/mira/sample_saved.txt')
    # sample_list = sample_load('data/local/mira/sample_saved.txt')

    # labelin
    AAE = list()
    PPE = list()
    for i in range(3, 4):
        print('K= ', i)
        labeler = Labeler(k=i)
        sample_list = labeler.label_samples(sample_list)
        # labeler.save('data/labeler_config.txt')

        # # run model
        model = SklearnRegressionModel(sample_list, labeler, 4, 1, raw_list)
        model.train()
        AAEtemp, PPEtemp = model.test()
        AAE.append(AAEtemp)
        PPE.append(PPEtemp)
    print(AAE)
    print(PPE)
    # model.save('linear_model.pkt')

    # # use model
    # print(model.predict(sample_list[0]))
