# from pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from matplotlib.pyplot import get
import numpy as np
import torch
from torch.utils.data import Subset

from neural_nets_training import params

MAX_PHYSICAL_BATCH_SIZE = 128


def get_uneven_data_loaders(dataset, distribution_counts):
    return torch.utils.data.random_split(dataset, distribution_counts)


def bubble_step(array, steps=3):
    """ lightly permute a array, by swapping neighboring values {steps} times
      randomly
    i.e.
    [0, 1, 2,3] -> [0, 2, 1,3] -> [2, 0, 1,3]
    """
    for i in range(steps):
        swap_ind = np.random.randint(len(array) - 1)
        temp = array[swap_ind]
        array[swap_ind] = array[swap_ind + 1]
        array[swap_ind + 1] = temp
    return array


def get_class_to_indices(dataset):
    """ returns a dictionary mapping class-id to a list of indices to only data from that class"""
    classes = np.arange(len(dataset.classes))
    class_to_dataset = {
        class_id: (np.array(dataset.targets) == class_id).nonzero()[0].T
        for class_id in classes
    }
    return class_to_dataset


def get_class_to_dataset(dataset):
    """ returns a dictionary mapping class-id to a dataset of only data from that class"""
    classes = np.arange(len(dataset.classes))
    #(dataset.targets[dataset.targets==class_id],dataset.data[dataset.targets==class_id] )
    class_to_dataset = {
        class_id: Subset(dataset, dataset.targets == class_id)
        for class_id in classes
    }
    return class_to_dataset


def zipf_dist(max_val, min_val, steps):
    """ returns a list of ints, where each int is a number of points from a class for a ZIPF distribution"""
    range_ = max_val - min_val
    return [int((range_ / (2**x)) + min_val) for x in range(steps)]


def get_zipf_class_distribution_counts(class_size,
                                       min_percentage=40,
                                       class_count=10):
    """ returns the number of points from each class according to a from zipf_distibution
    """
    percent = int(class_size / 100)
    class_distribution_counts = zipf_dist(int(class_size),
                                          min_percentage * percent,
                                          class_count)
    # the amount of data from each class in the ranking
    return class_distribution_counts


# Create a dictionary that has a dataset containing each class
def get_zipf_class_distribution_counts_from_dataset(dataset,
                                                    min_percentage=40):
    """ returns the number of points from each class accoridng to a from zipf_distibution
    
        returns list of ints (length = number of classes)
    """

    # get the smallest class size
    min_class_size = min((sum(np.array(dataset.targets) == i)
                          for i in range(len(dataset.classes))))

    return get_zipf_class_distribution_counts(class_size=min_class_size,
                                              min_percentage=min_percentage,
                                              class_count=len(dataset.classes))


def get_weighted_indices(*, class_ordering, class_distribution_counts,
                         class_to_dataset):
    """
    class_ordering : a list containing numbers 0-9 (inclusive), in some order
    distribution_counts : a list denoting the number of points to sample from a particular class
    class_to_dataset : a dictionary mapping class-id to a dataset of only data from that class

    returns a dictionary of lists of indices according to class_ordering and class_distribution_counts
    """
    indices = {}

    for i, class_id in enumerate(class_ordering):
        og_class_dataset = class_to_dataset[class_id]
        og_class_indices = class_to_dataset[class_id].indices.nonzero().T[0]

        number_of_points_from_class = int(
            min(class_distribution_counts[i], len(og_class_dataset)))
        # randomly pick a subset of the data points of a particular class
        indices[class_id] = np.random.choice(og_class_indices,
                                             size=number_of_points_from_class,
                                             replace=False)
    return indices


def generate_zipf_mask(dataset, min_percentage=40, reversed_zipf = False):
    """
    Generate a mask that is a zipf distribution
    # Note - this sort of re-implements get_weighted_indices, but I currently don't fully trust get_weighted_indices

    """
    N = len(dataset)
    class_count = len(dataset.classes)
    # returns a dictionary mapping class-id to indices of only data from that class
    class_to_indices = get_class_to_indices(dataset)
    # returns [ints]
    class_distribution_counts = get_zipf_class_distribution_counts_from_dataset(
        dataset, min_percentage=min_percentage)

    weighted_indices = []
    class_ordering = range(class_count)
    if reversed_zipf:
        class_ordering = reversed(class_ordering)
    for class_ind in class_ordering:
        # get the indices of the data points from a particular class
        class_indices = class_to_indices[class_ind]
        # randomly pick a subset of the data points of a particular class
        weighted_indices.extend(
            np.random.choice(class_indices,
                             size=class_distribution_counts[class_ind],
                             replace=False))
    print("concatenating mask")
    # construct a boolean mask for all the indices
    print(len(weighted_indices))
    mask = np.zeros(N, dtype=bool)
    mask[weighted_indices] = True
    return mask


def get_IID_datasets(dataset, num_teachers):
    """ 
    Function to create data loaders for the Teacher classifier 
    data is split IID
    """
    # data per teacher
    data_per_teacher = len(dataset) // num_teachers
    print(data_per_teacher)
    return [
        Subset(dataset,
               np.arange(0, data_per_teacher) + (data_per_teacher * teacher))
        for teacher in range(num_teachers)
    ]


def _get_IID_data_loaders(dataloader,
                          num_teachers,
                          batch_size=params.batch_size):
    """ Function to create data loaders for the Teacher classifier 
    
    data is split IID
    """
    # data per teacher
    data_per_teacher = len(dataloader) // num_teachers

    return [
        torch.utils.data.DataLoader(Subset(
            dataloader,
            np.arange(0, data_per_teacher) + (data_per_teacher * teacher)),
                                    batch_size=batch_size)
        for teacher in range(num_teachers)
    ]


def get_weighted_datasets_IID(dataset, number_of_datasets, weighted_indices):
    """ returns a set of datasets, which are sampled IID from the weighted dataset
    weighted_indices = {class : [list of indices]}
    """
    classes = len(weighted_indices)
    class_sizes = {
        key: len(indices)
        for key, indices in weighted_indices.items()
    }

    # number of samples per class, for a single teacher
    sample_counts = {
        key: int(class_size / number_of_datasets)
        for key, class_size in class_sizes.items()
    }

    def indices_for_teacher(dataset_ind):
        all_indices = []
        for class_ind in range(classes):
            all_indices.extend(weighted_indices[class_ind]
                               [dataset_ind *
                                sample_counts[class_ind]:(dataset_ind + 1) *
                                sample_counts[class_ind]])
        return all_indices

    return [
        Subset(dataset, indices_for_teacher(dataset_ind))
        for dataset_ind in range(number_of_datasets)
    ]


def _get_IID_data_loaders(dataloader, num_teachers):
    """ Function to create data loaders for the Teacher classifier 
    
    data is split IID
    """
    # data per teacher
    data_per_teacher = len(dataloader) // num_teachers

    return [
        torch.utils.data.DataLoader(Subset(
            dataloader,
            np.arange(0, data_per_teacher) + (data_per_teacher * teacher)),
                                    batch_size=params.batch_size)
        for teacher in range(num_teachers)
    ]


import random


def get_weighted_datasets_non_IID(dataset, number_of_datasets,
                                  weighted_indices):
    """ returns a set of datasets, which are sampled non- IID from the weighted dataset"""
    classes = len(weighted_indices)
    class_sizes = {
        key: len(indices)
        for key, indices in weighted_indices.items()
    }
    sample_counts = {
        key: int(class_size / number_of_datasets)
        for key, class_size in class_sizes.items()
    }

    def compute_zipf_distributions_indices_for_single_class(class_size):
        """ compute a zipf distribution
        I.e. according to a zipf distribution, create a list of lists
          such that the inner list represents a contiguous sequence of indices, such that the the 
          lengths of the sequences fall along a zipf dis (1/x) 
        """
        max_val = int(class_size / 4)
        min_val = int(class_size / 16)
        zipf = zipf_dist(max_val, min_val, number_of_datasets)
        zipf_sum = sum(zipf)
        #(min_val * number_of_datasets)
        rescaled_zipf = [
            int(class_size * dist_val / zipf_sum) for dist_val in zipf
        ]
        distributions = []
        running_sum = 0
        for val in rescaled_zipf:
            distributions.append(np.arange(running_sum, running_sum + val))
            running_sum += val
        #print([len(dist) for dist in distributions])
        random.shuffle(distributions)
        return distributions

    # a list of lists, such sample_counts[0] is the indices for a particular teacher and class_ind
    # sample_counts[class_ind][teacher_ind]
    sample_counts = [
        compute_zipf_distributions_indices_for_single_class(class_sizes[i])
        for i in range(classes)
    ]

    def indices_for_teacher(dataset_ind):
        all_indices = []
        for class_ind in range(classes):
            indicies_range = sample_counts[class_ind][dataset_ind]

            all_indices.extend(weighted_indices[class_ind][indicies_range])
        return all_indices

    return [
        Subset(dataset, indices_for_teacher(dataset_ind))
        for dataset_ind in range(number_of_datasets)
    ]
