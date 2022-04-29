from model.model import Model

class RegressionModel(Model):
    def __init__(self, sample_list, labeler):
        super().__init__(sample_list, labeler)
        pass

    # TODO
    def train(self):
        pass

    # TODO
    def predict(self, sample):
        pass

    # TODO
    def save(self, file_path):
        pass

    # TODO
    def load(self, file_path):
        pass