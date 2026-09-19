from src import config
from cloudlayer.factory import get_adapter


def main() -> None:
    cfg = config.load()
    adapter = get_adapter(cfg)

    job_id = adapter.submit_training(
        image_uri=cfg.training_image_uri,
        args={
            "n-estimators": 100,
            "max-depth": 4,
            "min-samples-leaf": 5,
            "seed": 20260101,
            "experiment": "itcs355-lab2",
            "run-name": "remote-train",
            "metrics-out": "/app/reports/remote-train.json",
        },
    )

    print(job_id)


if __name__ == "__main__":
    main()
