# DriftController.py
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
import torchvision

# custom packages
from src.config import config


class DriftController:
    def __init__(self, DatasetController):
        self.DatasetController = DatasetController
        self.source_class = None
        self.target_class = None
        self.drift_clients = None
        #记录drift过的class

    def enforce_drift(self, clients, drift_type, replace=True):
        if drift_type == 'class_swap':
            return self.sudden_swap_drift(clients, replace)
        elif drift_type == 'guass_noise':
            return self.guass_noise_drift(clients, replace)
        else:
            return self.no_drift(clients, replace)

    def no_drift(self, clients, replace):
        for client in clients:
            new_train_set, new_test_set = self.DatasetController.get_dataset_for_client(client)

            client.update_train(new_train_set, replace=replace)
            client.update_test(new_test_set, replace=True)

        message = 'No drift'

        return message

    def guass_noise_drift(self, clients, replace):
        if self.drift_clients is None:
            self.drift_clients = np.random.choice(len(clients), int(len(clients) * config.DRITF_PERCENTAGE), replace=False).tolist()

        for client in clients:
            new_train_set, new_test_set = self.DatasetController.get_dataset_for_client(client)
            if client.id in self.drift_clients:
                new_train_set.add_guassian_noise(1)
                new_test_set.add_guassian_noise(1)

            client.update_train(new_train_set, replace=replace)
            client.update_test(new_test_set, replace=True)

        message = 'Guassian noise is added to {} clients'.format(self.drift_clients)

        return message

    # constant drift
    # def constant_drift(self, percentage):
    #     drift_client = []
    #     source_class = np.random.randint(config.NUM_CLASS)
    #     while source_class in self.source_classes:
    #         source_class = np.random.randint(config.NUM_CLASS)
    #
    #     target_class = np.random.randint(config.NUM_CLASS)
    #     while target_class == source_class:
    #         target_class = np.random.randint(config.NUM_CLASS)
    #
    #     for client in self.clients:
    #         if client.distribution[source_class] > 0:
    #             drift_client.append(client.id)
    #             client.drift = True
    #
    #     # drift_client = np.random.choice(drift_client, int(len(drift_client) * percentage), replace=False)
    #
    #     self.drift_clients.append(drift_client)
    #     self.source_classes.append(source_class)
    #     self.target_classes.append(target_class)
    #     print('Drift Clients: {}. Source Class: {}. Target Class: {}'.format(self.drift_clients, self.source_classes,
    #                                                                          self.target_classes))

