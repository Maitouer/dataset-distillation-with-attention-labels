import logging
import os
import warnings
from dataclasses import dataclass

import pandas as pd
from datasets import disable_progress_bar, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from recbole.data import data_preparation

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

disable_progress_bar()


TASK_ATTRS = {
    # Amazon: All_Beauty
    "beauty": {
        "load_args": (
            "McAuley-Lab/Amazon-Reviews-2023",
            "0core_rating_only_All_Beauty",
        ),
    },
    # Amazon: Books
    "books": {
        "load_args": (
            "McAuley-Lab/Amazon-Reviews-2023",
            "0core_rating_only_Books",
        ),
    },
    # Amazon: Digital_Music
    "music": {
        "load_args": (
            "McAuley-Lab/Amazon-Reviews-2023",
            "0core_rating_only_Digital_Music",
        ),
    },
    # Amazon Magazine_Subscriptions
    "magazine": {
        "load_args": (
            "McAuley-Lab/Amazon-Reviews-2023",
            "0core_rating_only_Magazine_Subscriptions",
        ),
    },
}


@dataclass
class DataConfig:
    task_name: str
    datasets_path: str
    preprocessed_datasets_path: str
    train_batch_size: int = 32
    valid_batch_size: int = 256
    test_batch_size: int = 256
    model: str = "SASRec"
    recbole_config: str = "configs/sasrec.yaml"


class DataModule:
    """DataModule class
    ```
    data_module = DataModule(
        config.data,
    )
    # preprocess datasets
    data_module.run_preprocess(tokenizer=tokenizer)
    # preprocess external dataset (distilled data)
    data_module.preprocess_dataset(tokenizer=tokenizer, dataset=dataset)
    ```
    """

    def __init__(self, config):
        self.config = config
        # load raw dataset
        self.dataset_attr = TASK_ATTRS[self.config.task_name]
        self.datasets: DatasetDict = self.get_dataset()
        # preprocessed_dataset
        self.run_preprocess()
        # generate dataloader
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.get_dataloader()

        logger.info(f"Datasets: {self.datasets}")

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        else:
            assert self.config.task_name in TASK_ATTRS
            datasets = load_dataset(*self.dataset_attr["load_args"])

            os.makedirs(os.path.dirname(self.config.datasets_path), exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        return datasets

    def run_preprocess(self):
        """datasets preprocessing"""

        if os.path.exists(self.config.preprocessed_datasets_path):
            logger.info(
                "Load preprocessed datasets from `{}`".format(
                    self.config.preprocessed_datasets_path
                )
            )
            self.preprocessed_datasets = pd.read_csv(
                self.config.preprocessed_datasets_path
            )
            return

        self.preprocessed_datasets = self.preprocess_dataset(dataset=self.datasets)

        logger.info(
            f"Save preprocessed datasets to `{self.config.preprocessed_datasets_path}`"
        )
        # os.makedirs(
        #     os.path.dirname(self.config.preprocessed_datasets_path), exist_ok=True
        # )
        self.preprocessed_datasets.to_csv(
            self.config.preprocessed_datasets_path, sep="\t", index=False
        )

    def preprocess_dataset(self, dataset):
        dataset_df = pd.DataFrame(dataset["full"])
        dataset_df.columns = ["uid", "iid", "rating", "timestamp"]

        ### Filter users and items with less than 5 interactions ###
        filtered_review_df = dataset_df.groupby("iid").filter(lambda x: len(x) >= 3)
        filtered_review_df = (
            filtered_review_df.groupby("uid")
            .filter(lambda x: len(x) >= 5)
            .groupby("uid")
            .apply(
                lambda x: x.sort_values(by=["timestamp"], ascending=[True]),
                include_groups=True,
            )
            .reset_index(drop=True)
        )

        ### ID map ###
        unique_uids = filtered_review_df["uid"].unique()
        unique_iids = filtered_review_df["iid"].unique()
        uid_map = {old_id: new_id for new_id, old_id in enumerate(unique_uids)}
        iid_map = {old_id: new_id for new_id, old_id in enumerate(unique_iids)}
        mapped_review_df = filtered_review_df.copy()
        mapped_review_df["uid"] = mapped_review_df["uid"].map(uid_map)
        mapped_review_df["iid"] = mapped_review_df["iid"].map(iid_map)
        mapped_review_df.columns = [
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ]

        return mapped_review_df

    def get_dataloader(self):
        from recbole.config import Config
        from recbole.data.dataset import SequentialDataset

        # recbole config
        config = Config(
            model=self.config.model,
            dataset=self.config.task_name,
            config_file_list=[self.config.recbole_config],
        )
        config["dataset_name"] = self.config.task_name
        config["dataset_path"] = self.config.preprocessed_datasets_path
        config["train_batch_size"] = self.config.train_batch_size
        config["eval_batch_size"] = self.config.valid_batch_size
        config["test_batch_size"] = self.config.test_batch_size

        # dataset filtering
        self.datasets = SequentialDataset(config)
        # dataset splitting
        self.train_loader, self.valid_loader, self.test_loader = data_preparation(
            config, self.datasets
        )

    def train_loader(self):
        return self.train_loader

    def valid_loader(self):
        return self.valid_loader

    def test_loader(self):
        return self.test_loader
