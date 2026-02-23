import yaml
from train_script import train
from inference_script import inference

with open("experiment_configs/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)


train(config)
inference(config)
