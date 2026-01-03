import os

import pandas as pd
from datasets import DatasetDict
from flwr_datasets import FederatedDataset

class FixedDataFederatedDataset(FederatedDataset):
    def __init__(self, dataset: DatasetDict, dataset_name: str, data_dir_base: str, partitioners_dict: dict, preprocessing_pipeline=None):
        super().__init__(dataset=dataset_name, partitioners=partitioners_dict)
        # Override the dataset with the provided one. That works because the _dataset is evaluated lazily
        self._dataset = dataset
        self._dataset_prepared = True # That prevents trying to load the dataset again

        self.feature_columns = self.__get_feature_columns()
        self.preprocessing_pipeline = preprocessing_pipeline
        if self.preprocessing_pipeline is not None:
            print("Fitting preprocessing pipeline on train split.")
            self.__fit_preprocessing_pipeline_on_train_split()
            print("Done.")

        self.data_dir_base = data_dir_base

    def __get_feature_columns(self):
        train_split = self._dataset["train"].with_format("pandas")[:]
        feature_columns = [col for col in train_split.columns if not col in ["target", "label"]]

        return  feature_columns

    def __fit_preprocessing_pipeline_on_train_split(self):
        train_split = self._dataset["train"].with_format("pandas")[:]
        self.preprocessing_pipeline.fit(train_split[self.feature_columns])

    def __apply_preprocessing(self, df):

        if not self.preprocessing_pipeline is None:
            # check if pipeline has already been fitted
            if hasattr(self.preprocessing_pipeline, 'feature_names_in_'):
                print("Pipeline is fitted.")
            else:
                print("Pipeline is not fitted. Fitting on train split.")
                self.__fit_preprocessing_pipeline_on_train_split()

            df[self.feature_columns] = self.preprocessing_pipeline.transform(df[self.feature_columns])
        return df

    def __construct_partition_save_path(self, partition_id: int):
        return os.path.join(self.data_dir_base, 'partition_' + str(partition_id), 'data.csv')

    def save_partition_as_csv(self, partition_id: int):

        path = self.__construct_partition_save_path(partition_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        partition_df = super().load_partition(partition_id, "train").with_format("pandas")[:]
        partition_df = self.__apply_preprocessing(partition_df)

        partition_df.to_csv(path, index=False)

    def load_partition(self, partition_id: int, split: str):
        #partition_df = super().load_partition(partition_id, split).with_format("pandas")[:]
        path = self.__construct_partition_save_path(partition_id)
        # check if file exists
        if not os.path.exists(path):
            self.save_partition_as_csv(partition_id)
        partition_df = pd.read_csv(path) # this is already preprocessed
        return partition_df

    def save_test_split_as_csv(self):
        path = self.__construct_partition_save_path("test")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.load_test_split().to_csv(path, index=False)

    def save_train_split_as_csv(self):

        path = self.__construct_partition_save_path("train")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.load_train_split().to_csv(path, index=False)

    def load_test_split(self):

        path = self.__construct_partition_save_path("test")
        if not os.path.exists(path):
            test_df = self._dataset["test"].with_format("pandas")[:]
            test_df = self.__apply_preprocessing(test_df)
            test_df.to_csv(path, index=False)
        return pd.read_csv(path)

    def load_train_split(self):

        path = self.__construct_partition_save_path("train")
        if not os.path.exists(path):
            train_df = self._dataset["train"].with_format("pandas")[:]
            train_df = self.__apply_preprocessing(train_df)
            train_df.to_csv(path, index=False)
        return pd.read_csv(path)