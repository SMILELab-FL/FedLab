import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('../../../')

from fedlab.utils.dataset.sampler import RawPartitionSampler

from fedlab_benchmarks.models.cnn import CNN_Cifar10, CNN_Femnist, CNN_Mnist
from fedlab_benchmarks.models.rnn import RNN_Shakespeare
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataloader
from fedlab_benchmarks.datasets.leaf_data_process.nlp_utils.dataset_vocab.sample_build_vocab import get_built_vocab


def get_dataset(args):
    if args.dataset == 'mnist':
        root = '../../../datasets/data/mnist/'
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,
                                             transform=test_transform)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=RawPartitionSampler(trainset,
                                        client_id=args.rank,
                                        num_replicas=args.world_size - 1),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.world_size)

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=int(
                                                     len(testset) / 10),
                                                 drop_last=False,
                                                 num_workers=2,
                                                 shuffle=False)
    elif args.dataset == 'femnist':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    elif args.dataset == 'shakespeare':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    else:
        raise ValueError("Invalid dataset:", args.dataset)

    return trainloader, testloader


def get_model(args):
    if args.dataset == "mnist":
        model = CNN_Mnist()
    elif args.dataset == 'femnist':
        model = CNN_Femnist()
    elif args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
    else:
        raise ValueError("Invalid dataset:", args.dataset)
        
    return model