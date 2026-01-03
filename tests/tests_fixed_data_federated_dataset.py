# python
import os

import numpy as np
from datasets import Dataset, DatasetDict
import flwr_datasets
import pytest

from fed_fuzzy_clust_regression.fixed_data_federated_dataset import FixedDataFederatedDataset


def _make_random_dataset(n_train=50, n_test=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    train = {
        "x": rng.standard_normal((n_train,)).tolist(),
        "label": rng.integers(0, n_classes, size=(n_train,)).tolist(),
    }
    test = {
        "x": rng.standard_normal((n_test,)).tolist(),
        "label": rng.integers(0, n_classes, size=(n_test,)).tolist(),
    }
    return DatasetDict({"train": Dataset.from_dict(train), "test": Dataset.from_dict(test)})


def test_init_stores_dataset_and_load_partition():
    ds = _make_random_dataset()

    dataset_name = "my_dataset"
    partitioners = {"train": flwr_datasets.partitioner.IidPartitioner(num_partitions=5)}

    inst = FixedDataFederatedDataset(ds, dataset_name, partitioners)

    # dataset stored and prepared flag set
    assert hasattr(inst, "_dataset")
    assert inst._dataset is ds
    assert getattr(inst, "_dataset_prepared") is True

    # ensure base __init__ was called and received expected kwargs
    assert inst._dataset_name == dataset_name
    assert inst._partitioners == partitioners
    # still an instance of the real base class
    assert isinstance(inst, flwr_datasets.FederatedDataset)

    # load partitions and check data
    for partition_id in range(5):
        partition = inst.load_partition(partition_id)
        assert partition.num_rows > 5
        assert partition.num_columns < 15


def test_write_csv_method():
    ds = _make_random_dataset()
    dataset_name = "my_dataset"
    partitioners = {"train": flwr_datasets.partitioner.IidPartitioner(num_partitions=5)}

    inst = FixedDataFederatedDataset(ds, dataset_name, partitioners)

    # create files
    for partition_id in range(5):
        path = f"./tmp/test_partition_{partition_id}.csv"
        inst.save_partition_as_csv(partition_id=partition_id, path=path)

    # check files were created
    for partition_id in range(5):
        path = f"./tmp/test_partition_{partition_id}.csv"
        assert os.path.exists(path)

    # Cleanup
    for partition_id in range(5):
        os.remove(f"./tmp/test_partition_{partition_id}.csv")
