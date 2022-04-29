# 这只是一个main文件示例，里面应该写上标准运行过程。如果你只需要在本地测试模块的功能却又不需要上传，就不要修改main.py。
# 如果需要添加临时测试脚本或数据，将它们以tmp开头命名。例如tmp_test.py。这些文件不会加入git中。
# data文件夹下的文件会上传github，不要直接存储自己的数据。可以数据存在data/local文件夹下。
# 可以添加新函数，但不要修改任何api。如果有结构性问题，确保所有人同意再修改。
# 如果你认为有些函数或功能不止在你的代码会用到，考虑把它添加到utils文件夹下。
# 原则上不要添加任何全局变量。如果实在需要，考虑重构成class。
# 使用符合python的命名规则。https://zhuanlan.zhihu.com/p/423130392

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