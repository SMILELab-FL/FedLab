cifar10_iid_baseline_config = {
    "partition": "iid",
    "round": 4000,
    "network": "alexnet",
    "sample_ratio": 0.1,
    "dataset": "cifar10",
    "total_client_num": 100,
    "lr": 0.1,
    "batch_size": 100,
    "epochs": 5
}

cifar10_noniid_baseline_config = {
    "partition": "noniid",
    "round": 4000,
    "network": "alexnet",
    "sample_ratio": 0.1,
    "dataset": "cifar10",
    "total_client_num": 100,
    "lr": 0.1,
    "batch_size": 100,
    "epochs": 5
}