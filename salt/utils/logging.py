import os

from pytorch_lightning.loggers import CometLogger


def get_comet_logger(experiment_name):
    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace=os.environ["COMET_WORKSPACE"],
        project_name="salt",
        experiment_name=experiment_name,
        # experiment_key=exp_id,
    )
    return logger
