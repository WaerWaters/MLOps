import mlflow


def promote_if_best(model_name, new_version, accuracy):
    client = mlflow.MlflowClient()

    production_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not production_versions:
        client.transition_model_version_stage(
            name=model_name, version=new_version, stage="Production"
        )
        mlflow.set_tag("deployed", "true")
        print("No existing production model — new model promoted to Production")
    else:
        prod_run = client.get_run(production_versions[0].run_id)
        prod_accuracy = prod_run.data.metrics["inference accuracy"]

        if accuracy > prod_accuracy:
            client.transition_model_version_stage(
                name=model_name,
                version=production_versions[0].version,
                stage="Archived",
            )
            client.transition_model_version_stage(
                name=model_name, version=new_version, stage="Production"
            )
            mlflow.set_tag("deployed", "true")
            print(
                f"New model outperforms production ({accuracy:.4f} > {prod_accuracy:.4f}) — promoted to Production"
            )
        else:
            mlflow.set_tag("deployed", "false")
            print(
                f"New model does not outperform production ({accuracy:.4f} <= {prod_accuracy:.4f}) — not promoted"
            )
