from datasets import load_dataset
from transformers import AutoImageProcessor

class Data():
    def __init__(self): 
        self.data = load_dataset("zh-plus/tiny-imagenet")
        self.preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")

    def get_preprocessed(self, inputs):
        inputs_processed = self.preprocessor(inputs["image"], return_tensors="pt")
        inputs["pixel_values"] = inputs_processed["pixel_values"]
        return inputs