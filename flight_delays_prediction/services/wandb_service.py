from datetime import datetime

import wandb
from wandb.wandb_run import Run
from wandb.sdk import login


class WandbService:
    """
    Custom context manager to connect to Weights and Biases
    """

    def __init__(self, api_key, project_name, job_type):
        self.project_name = project_name
        self.job_type = job_type
        self.run: Run | None = None
        self.experiment_start_date = str(
            datetime.today().date().strftime(format="%Y-%m-%d")
        )
        self.__connect(api_key=api_key)

    def __connect(self, api_key):
        login(key=api_key)

    def __enter__(self) -> Run:
        self.run = wandb.init(
            project="flight-delay-prediction",
            group="Experiment" + str(self.experiment_start_date),
            notes="Experiment created on " + str(self.experiment_start_date),
        )
        return self.run

    def __exit__(self, exc_type, exc_value, traceback):
        self.run.finish()
