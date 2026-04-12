import yaml
import mlflow
import sys

# first argument is git hash
git_hash = sys.argv[1]

with open("experiment_configs/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)


mlflow_server = "http://172.24.198.42:5050"
mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment("dvml8_gruppe1")


# train(config, git_hash)
# inference(config, git_hash)

client = mlflow.MlflowClient()
client.transition_model_version_stage("dvml_gruppe1", "2", stage="None")
