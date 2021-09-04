# Modify from Flower's spilt_json_data.py for shakespeare data in leaf
# [https://github.com/adap/flower/blob/main/baselines/flwr_baselines/scripts/leaf/shakespeare/split_json_data.py]

"""Splits LEAF generated datasets and creates individual client partitions."""
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

sys.path.append("../../../../")

from fedlab_benchmarks.datasets.leaf_data_process.dataset.shakespeare_dataset import ShakespeareDataset


def save_dataset_pickle(save_root: Path, user_idx: int, dataset_type: str, dataset):
    """Saves partition for specific client
    Args:
        save_root (Path): Root folder where to save partition
        user_idx (int): User ID
        dataset_type (str): Dataset type {train, test}
        dataset (ShakespeareDataset): Dataset {train, test}
    """
    save_dir = save_root / dataset_type
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{dataset_type}_{str(user_idx)}.pickle", "wb") as save_file:
        pickle.dump(dataset, save_file)


def process_user(
        json_file: Dict[str, Any],
        user_idx: str,
        user_str: str,
        dataset_type: str,
        save_root: Path,
):
    """Creates and saves partition for user
    Args:
        json_file (Dict[str, Any]): JSON file containing user data
        user_idx (str): User ID (counter) in string format
        user_str (str): Original User ID
        dataset_type (str): Dataset type {train, test}
        save_root (Path): Root folder where to save the partition
    """
    sentence = json_file["user_data"][user_str]["x"]
    next_char = json_file["user_data"][user_str]["y"]
    dataset = ShakespeareDataset(client_id=user_idx, client_str=user_str,
                                 input=sentence, output=next_char)
    save_dataset_pickle(save_root, user_idx, dataset_type, dataset)


def split_json_and_save(
        dataset_type: str,
        paths_to_json: List[Path],
        save_root: Path,
):
    """Splits LEAF generated datasets and creates individual client partitions.
    Args:
        dataset_type (str): Dataset type {train, test}
        paths_to_json (PathLike): Path to LEAF JSON files containing dataset.
        save_root (Path): Root directory where to save the individual client
            partition files.
    """
    user_count = 0
    for path_to_json in paths_to_json:
        with open(path_to_json, "r") as json_file:
            json_file = json.load(json_file)
            users_list = sorted(json_file["users"])
            num_users = len(users_list)
            for user_idx, user_str in enumerate(users_list):
                process_user(
                    json_file, user_count + user_idx, user_str, dataset_type, save_root
                )
        user_count += num_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""download and process a LEAF Shakespeare train/test dataset,
        save each client's train/test dataset in their respective folder in a form of pickle."""
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="""Root folder where partitions will be save as
                {save_root}/{train,test}/{client_id}.pickle""",
    )
    parser.add_argument(
        "--leaf_train_jsons_root",
        type=str,
        required=True,
        help="""Complete path to JSON file containing the generated
                trainset for LEAF Shakespeare.""",
    )
    parser.add_argument(
        "--leaf_test_jsons_root",
        type=str,
        required=True,
        help="""Complete path to JSON file containing the generated
            *testset* for LEAF Shakespeare.""",
    )

    args = parser.parse_args()

    # Split train dataset into train and validation
    # then save files for each client
    original_train_datasets = sorted(
        list(Path(args.leaf_train_jsons_root).glob("**/*.json"))
    )
    # original_train_dataset = Path(args.leaf_train_json)
    existing_users = split_json_and_save(
        dataset_type="train",
        paths_to_json=original_train_datasets,
        save_root=Path(args.save_root),
    )

    # Split and save the test files
    original_test_datasets = sorted(
        list(Path(args.leaf_test_jsons_root).glob("**/*.json"))
    )
    # original_test_dataset = Path(args.leaf_test_json)
    split_json_and_save(
        dataset_type="test",
        paths_to_json=original_test_datasets,
        save_root=Path(args.save_root),
    )
