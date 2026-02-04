from transformers import AutoConfig, AutoModelForImageClassification

class ImageModel():
    def __init__(self):
        config = AutoConfig.from_pretrained("google/mobilenet_v2_1.0_224", num_labels=200)
        self.model = AutoModelForImageClassification.from_config(config)

    def describe(self):
        print(self.model.config)