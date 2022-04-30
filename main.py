from model.linear_regression import RegressionModel
from preprocess.preprocess_theta import PreprocessorTheta
from preprocess.Preprocessor_mira import PreprocessorMira
from preprocess.preprocess_taiyi import PreprocessorTaiyi
from sample.sample import sample_save, sample_load, to_sample_list
from sample.class_labeling import Labeler


if __name__ == '__main__':
    # preprocess
    preprocessor = PreprocessorMira()
    raw_list = preprocessor.preprocess('data/local/mira/sorted.csv')
    preprocessor.save(raw_list,'data/local/mira/RawSample_saved.txt')
    # raw_list = preprocessor.load('data/local/theta/RawSample_saved.txt')


    # generate sample
    sample_list = to_sample_list(raw_list)
    sample_save(sample_list, 'data/local/mira/sample_saved.txt')
    # sample_list = sample_load('data/local/theta/sample_saved.txt')

    # labeling
    labeler = Labeler()
    sample_list = labeler.label_samples(sample_list)
    # labeler.save('data/labeler_config.txt')


    # # run model
    # model = RegressionModel(sample_list, labeler)
    # model.train()
    # model.save('linear_model.pkt')

    # # use model
    # print(model.predict(sample_list[0]))