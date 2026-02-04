from datasets import load_from_disk, concatenate_datasets
from transformers import AutoImageProcessor
import torch

class Data():
    def __init__(self, path): 
        self.data = load_from_disk(path)
        self.preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224", use_fast = True)

    def get_train_val_test_sets(self, splits: list[float]):
        combined = concatenate_datasets([self.data["train"], self.data["valid"]])
        
        total = len(combined)
        train_size = int(total * splits[0])
        val_size = int(total * splits[1])

        combined = combined.shuffle(seed=1)

        return {
            "train": combined.select(range(train_size)),
            "val": combined.select(range(train_size, train_size + val_size)),
            "test": combined.select(range(train_size + val_size, total))}

    def collate_fn(self, batch):
        '''
        This assures a 3 dimensional channel.
        Initial assumption was that the images already were exclusively RGB,
        which was wrong as some images are also Grayscale (1 dim) - this made the preprocessor break.

        The preprocessor assures that all Images are 224x224, which is what the chosen model expects.
        Our chosen dataset consists of exclusively 64x64 images, meaning that they are all upscaled.
        '''
        images = [x["image"].convert("RGB") for x in batch]
        labels = [x["label"] for x in batch]
        inputs = self.preprocessor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    def save_to_disk(self, path):
        self.data.save_to_disk(path)