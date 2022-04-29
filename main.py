from model.linear_regression import RegressionModel
from preprocess.preprocess_theta import PreprocessorTaiyi
from sample.sample import sample_save, sample_load, to_sample_list
from sample.class_labeling import Labeler


if __name__ == '__main__':
    # preprocess
    preprocessor = PreprocessorTaiyi()
    raw_list = preprocessor.preprocess('data/taiyi.dat')
    preprocessor.save(raw_list, 'data/taiyi_raw_saved.txt')

    # generate sample
    sample_list = to_sample_list(raw_list)
    sample_save(sample_list, 'data/sample_saved.txt')

    # labeling
    labeler = Labeler()
    sample_list = labeler.label_samples(sample_list)
    labeler.save('data/labeler_config.txt')

    # run model
    model = RegressionModel(sample_list, labeler)
    model.train()
    model.save('linear_model.pkt')

    # use model
    print(model.predict(sample_list[0]))