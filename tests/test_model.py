import os
import yaml
from train_script import train
from inference_script import inference


def test_train_saves_model():
    with open("experiment_configs/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["for_testing"] = True
    train(config, git_hash="test")
    assert os.path.exists(config["save_path"])


def test_inference_runs():
    with open("experiment_configs/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["for_testing"] = True
    avg_loss, accuracy = inference(config, git_hash="test")
    assert avg_loss >= 0
    assert 0 <= accuracy <= 1
