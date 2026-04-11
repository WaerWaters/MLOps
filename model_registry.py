import mlflow


def promote_if_best(model_name, new_version, accuracy, git_hash="unknown"):
    client = mlflow.MlflowClient()

    # Tag the new version with the git commit hash for traceability
    client.set_model_version_tag(model_name, new_version, "git_commit", git_hash)

    # Check if there is a current production model via alias
    try:
        production_version = client.get_model_version_by_alias(model_name, "production")
        prod_run = client.get_run(production_version.run_id)
        prod_accuracy = prod_run.data.metrics["inference accuracy"]

        if accuracy > prod_accuracy:
            client.set_registered_model_alias(model_name, "production", new_version)
            mlflow.set_tag("deployed", "true")
            print(
                f"New model outperforms production ({accuracy:.4f} > {prod_accuracy:.4f}) — promoted to production"
            )
        else:
            mlflow.set_tag("deployed", "false")
            print(
                f"New model does not outperform production ({accuracy:.4f} <= {prod_accuracy:.4f}) — not promoted"
            )

    except mlflow.exceptions.MlflowException:
        # No production alias exists yet, promote directly
        client.set_registered_model_alias(model_name, "production", new_version)
        mlflow.set_tag("deployed", "true")
        print("No existing production model — new model promoted to production")
