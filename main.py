from models.image_classifier import ImageModel
from data.get_data import Data


# model = ImageModel()
# model.describe()

data = Data()
preprocessed = data.get_preprocessed(data.data["train"][0])
print(preprocessed["pixel_values"])

2+2
