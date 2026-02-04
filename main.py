from models.image_classifier import ImageModel
from data.get_data import Data


# model = ImageModel()
# model.describe()

data = Data(path="data/tiny-imagenet")

data_splits = data.get_train_val_test_sets(splits=[0.7, 0.1, 0.2])
# preprocessed = data.get_preprocessed(data.data["train"][0])
# print(preprocessed["pixel_values"])