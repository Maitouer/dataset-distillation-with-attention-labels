import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
import mlflow
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig, Evaluator
from hydra.core.config_store import ConfigStore
from model import ModelConfig, SASRec
from omegaconf import OmegaConf
from recbole.config import Config as RecBoleConfig
from tqdm.contrib.logging import logging_redirect_tqdm
from trainer import TrainConfig, Trainer
from transformers import set_seed
from utils import log_params_from_omegaconf_dict

from data import DataConfig, DataModule

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add main.log
            mlflow.log_artifact(os.path.join(output_dir, "main.log"))
        return out

    return wrapper


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@mlflow_start_run_with_hydra
def main(config: Config):
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # Set seed
    set_seed(config.base.seed)

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data)

    # Learner
    logger.info(f"Building leaner model: (`{config.model.model_name}`)")

    # recbole config
    recbole_config = RecBoleConfig(
        model=config.model.model_name,
        dataset=config.data.task_name,
        config_file_list=[config.data.recbole_config],
        config_dict=config.model,
    )
    model = SASRec(recbole_config, data_module.datasets)

    # Distilled data
    if config.distilled_data.pretrained_data_path is not None:
        distilled_data = DistilledData.from_pretrained(
            config.distilled_data.pretrained_data_path
        )
    else:
        distilled_data = DistilledData(
            config=config.distilled_data,
            train_config=config.learner_train,
            seq_num=data_module.datasets.user_num,
            seq_length=data_module.datasets.max_item_list_len,
            num_items=data_module.datasets.item_num,
            num_layers=model.n_layers,
            num_heads=model.n_heads,
        )
    logger.info(
        f"Distilled data shape: (`{list(distilled_data.inputs_embeds.data.shape)}`)"
    )

    # Evaluator
    evaluator = Evaluator(config.evaluate, recbole_config, model=model)

    # Train distilled data
    if not config.train.skip_train:
        trainer = Trainer(config.train)
        trainer.fit(
            distilled_data=distilled_data,
            model=model,
            train_loader=data_module.train_loader,
            valid_loader=data_module.valid_loader,
            evaluator=evaluator,
        )

    # Evaluate
    results = evaluator.evaluate(
        distilled_data, eval_loader=data_module.valid_loader, verbose=False
    )
    mlflow.log_metrics({f"avg.{k}": v[0] for k, v in results.items()})
    mlflow.log_metrics({f"std.{k}": v[1] for k, v in results.items()})

    results = {k: f"{v[0]}±{v[1]}" for k, v in results.items()}
    logger.info(f"Final Results: {results}")
    if not os.path.exists(config.base.save_dir):
        os.mkdir(config.base.save_dir)
    save_path = os.path.join(config.base.save_dir, "results.json") 
    json.dump(results, open(save_path, "w"))
    mlflow.log_artifact(save_path)

    return


if __name__ == "__main__":
    main()
