import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
import torchvision
from utils.CustomDatasets import load_CINIC10

logger = logging.getLogger(__name__)


DATA_PATH = './data/'
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, batch_size=128, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    def __add__(self, other):
        assert self.tensors[0][0].size(0) == other.tensors[0][0].size(0)
        data = torch.cat((self.tensors[0], other.tensors[0]), 0)
        label = torch.cat((self.tensors[1], other.tensors[1]), 0)

        self.tensors = (data, label)

    def sort_by_class(self):
        sorted_dataset_by_class = []
        images = self.tensors[0].numpy()
        images = images.reshape((images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))
        labels = self.tensors[1].numpy()

        for i in np.unique(labels):
            idx = np.where(labels == i)
            sorted_dataset_by_class.append(images[idx, :][0])

        return sorted_dataset_by_class

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True)

#替换类别，class1，2对应从X替换成X，会替换所有client
    def class_swap(self, class_1, class_2):
        labels = self.tensors[1].numpy()
        class_1_labels = np.array(labels == class_1).astype(int)
        # 注释为类别交换
        #class_2_labels = np.array(labels == class_2).astype(int)
# 注释为类别交换
        class_1_labels = class_1_labels * (class_2 - class_1)
        #class_2_labels = class_2_labels * (class_1 - class_2)

        labels += class_1_labels #+ class_2_labels #也需要
        self.tensors = (self.tensors[0], torch.Tensor(labels))

    class AddGaussianNoise(object):
        def __init__(self, level=1):
            self.std = level

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def add_guassian_noise(self, level):
        images = self.tensors[0]
        images = images + torch.randn(images.size()) * level
        self.tensors = (images, self.tensors[1])


# 负责根据各种分布生成训练集
class DatasetController:
    def __init__(self, dataset_name, number_of_training_samples, number_of_testing_samples):
        global training_dataset, test_dataset, transform
        self.number_of_training_samples = number_of_training_samples
        self.number_of_testing_samples = number_of_testing_samples

        dataset_name = dataset_name.upper()
        # get dataset from torchvision.datasets if exists
        if hasattr(torchvision.datasets, dataset_name) or dataset_name == 'FASHIONMNIST':
            # set transformation differently per dataset
            if dataset_name == "CIFAR10":
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5071, 0.4867, 0.4408))
                    ]
                )
            elif dataset_name == "CIFAR100":
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                    ]
                )
            elif dataset_name in ["MNIST", "FASHIONMNIST"]:
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5,), (0.5,))  # Normalize Fashion MNIST
                    ]
                )
                dataset_name = 'FashionMNIST' if dataset_name == "FASHIONMNIST" else dataset_name

            # prepare raw training & test datasets
            training_dataset = torchvision.datasets.__dict__[dataset_name](
                root=DATA_PATH,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = torchvision.datasets.__dict__[dataset_name](
                root=DATA_PATH,
                train=False,
                download=True,
                transform=transform
            )

        else:
            if dataset_name == 'CINIC10':
                training_dataset, test_dataset, transform = load_CINIC10(DATA_PATH)
            else:
                # dataset not found exception
                error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
                raise AttributeError(error_message)

        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        if test_dataset.data.ndim == 3:
            test_dataset.data.unsqueeze_(3)
        self.num_class = np.unique(training_dataset.targets).shape[0]

        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
            test_dataset.data = np.asarray(test_dataset.data)

        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()
            test_dataset.targets = test_dataset.targets.tolist()

        sorted_train_data, sorted_train_indices = self.get_sorted_data_and_indices(training_dataset.data, training_dataset.targets)
        sorted_test_data, sorted_test_indices = self.get_sorted_data_and_indices(test_dataset.data, test_dataset.targets)

        self.sorted_train_data = sorted_train_data
        self.sorted_train_indices = sorted_train_indices
        self.sorted_test_data = sorted_test_data
        self.sorted_test_indices = sorted_test_indices
        self.transform = transform

    def get_sorted_data_and_indices(self, data, targets):
        from collections import defaultdict
        sort_dict = defaultdict(list)

        for i, target in enumerate(targets):
            sort_dict[target].append(i)

        sorted_targets = []
        sorted_indices = []

        for key in sorted(list(sort_dict.keys())):
            sorted_targets.extend([key] * len(sort_dict[key]))
            sorted_indices.extend(sort_dict[key])

        data = data[sorted_indices]

        dataset_indices = []
        start = 0
        for count in torch.bincount(torch.tensor(sorted_targets)):
            dataset_indices.append(np.arange(start, start + count))
            start += count

        return data, dataset_indices

    def draw_data_by_distribution(self, distribution, total_samples, train=True, remove_from_pool=True, draw_from_pool=True):
        distribution = distribution * total_samples
        new_input = []
        new_label = []
        for class_id, n in enumerate(distribution):
            selected_indices = self.draw_data_index_by_class(class_id, n, train, remove_from_pool, draw_from_pool)
            if train:
                new_input.extend(self.sorted_train_data[selected_indices])
            else:
                new_input.extend(self.sorted_test_data[selected_indices])

            extra = max(int(n) - len(selected_indices), 0)
            if extra > 0:
                extra_indices = self.draw_data_index_by_class(class_id, extra, remove_from_pool=False,
                                                              draw_from_pool=False)
                if train:
                    new_input.extend(self.sorted_train_data[extra_indices])
                else:
                    new_input.extend(self.sorted_test_data[extra_indices])

            for _ in range(int(n)):
                new_label.append(class_id)

        new_dataset = CustomTensorDataset((torch.Tensor(np.array(new_input)), torch.Tensor(new_label)),
                                          transform=self.transform)

        return new_dataset

    def draw_data_index_by_class(self, class_id, n, train=True, remove_from_pool=True, draw_from_pool=True):
        if train:
            available_indices = self.sorted_train_indices[class_id]
        else:
            available_indices = self.sorted_test_indices[class_id]

        if draw_from_pool:
            indices = np.arange(1, len(available_indices) - 1)
            selected_indices = np.random.choice(indices, min(len(indices), int(n)), replace=False)
            res = available_indices[selected_indices]
        else:
            indices = np.arange(available_indices[0], available_indices[-1] + 1)
            res = np.random.choice(indices, int(n), replace=False)

        if remove_from_pool:
            if train:
                self.sorted_train_indices[class_id] = np.delete(available_indices, selected_indices)
            else:
                self.sorted_test_indices[class_id] = np.delete(available_indices, selected_indices)

        return res

    def get_dataset_for_client(self, client):
        new_train_set = self.draw_data_by_distribution(client.distribution, self.number_of_training_samples, train=True)

        new_test_set = self.draw_data_by_distribution(client.distribution, self.number_of_testing_samples, train=False,
                                                          remove_from_pool=False, draw_from_pool=False)

        return new_train_set, new_test_set


def get_dataloader(data):
    return DataLoader(data, batch_size=data.batch_size, shuffle=True)