import copy
import logging

import numpy as np
import torch


def get_sampler(dataset, tasks, **kwargs):#capacity=None):
    if "omni" in dataset:
        return OmniglotSampler(tasks, **kwargs)


class OmniglotSampler:
    # Class to sample tasks
    def __init__(self, tasks, trainset, testset,
                 batch_size, validation_batch_size, all_tasks_batch_size):
        self.tasks = tasks
        self.task_sampler = SampleOmni(
            trainset, testset,
            batch_size=batch_size, validation_batch_size=validation_batch_size,
            all_tasks_batch_size=all_tasks_batch_size)

        self.task_sampler.add_complete_iterator(self.tasks)

    def sample_all_tasks(self):
        return self.task_sampler.complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)


class OmniglotSamplerImproved:
    # Class to sample tasks
    def __init__(self, tasks, trainset, testset,
                 batch_size, validation_batch_size, all_tasks_batch_size):
        self.tasks = tasks
        self.task_sampler = SampleOmni(
            trainset, testset,
            batch_size=batch_size, validation_batch_size=validation_batch_size,
            all_tasks_batch_size=all_tasks_batch_size)

        self.task_sampler.add_complete_iterator(self.tasks)

    def sample_all_tasks(self):
        return self.task_sampler.complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)


class SampleOmni:
    def __init__(self, trainset, testset, batch_size, validation_batch_size,
                 all_tasks_batch_size):
        self.task_train_iterators = []
        self.trainset = trainset
        self.testset = testset

        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.all_tasks_batch_size = all_tasks_batch_size
        self.iterators = {}
        self.validation_iterators = {}


    def add_complete_iterator(self, tasks):
        dataset = self.get_task_trainset(tasks, True)
        train_iterator = torch.utils.data.DataLoader(dataset,
            batch_size=self.all_tasks_batch_size, shuffle=True, num_workers=1)
        self.complete_iterator = train_iterator
        print("Len of complete iterator = %d", len(self.complete_iterator) * 64)

    def add_task_iterator(self, task, train):
        dataset = self.get_task_trainset([task], train)
        if train:
            iterator = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
            self.iterators[task] = iterator
        else:
            iterator = torch.utils.data.DataLoader(
                dataset, batch_size=self.validation_batch_size, shuffle=True, num_workers=1)
            self.validation_iterators[task] = iterator
        print("Task %d has been added to the list" % task)
        return iterator

    def get(self, tasks, train):
        if train:
            for task in tasks:
                if task in self.iterators:
                    return self.iterators[task]
                else:
                    return self.add_task_iterator(task, True)
        else:
            for task in tasks:
                if task in self.validation_iterators:
                    return self.validation_iterators[task]
                else:
                    return self.add_task_iterator(task, False)


    def get_task_trainset(self, task, train):
        if train:
            trainset = copy.deepcopy(self.trainset) # getting train/test sets
        else:
            trainset = copy.deepcopy(self.testset)

        class_labels = np.array([x[1] for x in trainset._flat_character_images])

        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]

        return trainset

    def get_task_testset(self, task):
        trainset = copy.deepcopy(self.testset)
        class_labels = np.array([x[1] for x in trainset._flat_character_images])

        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]

        return trainset

    def filter_upto(self, task):
        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]
        return trainset
