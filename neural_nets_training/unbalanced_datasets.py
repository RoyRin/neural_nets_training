# from pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import copy
import random

import numpy as np
import torch
from torch.utils.data import Subset

from neural_nets_training import params

MAX_PHYSICAL_BATCH_SIZE = 128


def get_uneven_data_loaders(dataset, distribution_counts):
    return torch.utils.data.random_split(dataset, distribution_counts)


def get_class_to_indices(dataset):
    """ returns a dictionary mapping class-id to a list of 
    indices to only data from that class
    
    """
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

def compute_entropy(probs):
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log(probs))



def get_distributions(dataset, teacher_indices):
    """Get the distribution of classes for each teacher
    (this is used a sanity check that the teacher_indices are correct)

    Args:
        teacher_indices (_type_): _description_

    Returns:
        _type_: _description_
    """
    dists = []
    for indices in teacher_indices:# .items():
        dist = np.zeros(len(dataset.classes))
        for index in indices:
            dist[dataset.targets[index]] += 1
        dists.append(dist)
    return dists



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
        example: [5421, 3790, 2975, 2567, 2363, 2261, 2210, 2185, 2172, 2166]
    """

    # get the smallest class size
    min_class_size = min((sum(np.array(dataset.targets) == i)
                          for i in range(len(dataset.classes))))

    return get_zipf_class_distribution_counts(class_size=min_class_size,
                                              min_percentage=min_percentage,
                                              class_count=len(dataset.classes))


def get_indices_by_class_size(*, class_ordering, class_distribution_counts,
                         class_to_dataset):
    """
    Get a dictionary of [indices] for each class, such that each class has class_distribution_counts[class_ind] # of indices

    Args:
        class_ordering : a list containing numbers 0-9 (inclusive), in some order
        distribution_counts : a list denoting the number of points to sample from a particular class
        class_to_dataset : a dictionary mapping class-id to a dataset of only data from that class
    Returns:
        a dictionary of lists of indices
        {class_id: [Indices]} such that each class has the assigned number of indices
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
    # Note - this sort of re-implements get_indices_by_class_size, but I currently don't fully trust get_indices_by_class_size

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

###

def generate_non_uniform_probability_of_drawing(teacher_count, class_count):
    """Create a probability distribution for the teacher to draw from each class in a non-uniform way

    Args:
        teacher_count (_type_): _description_
        class_count (_type_): _description_

    Returns:
        _type_: _description_
    """
    teachers_probabilities = []
    for i in range(teacher_count):
        probabilities = [ 
            (100 * ((1/2)** x)) for x in range(class_count)
        ] # zipf distribution
        probabilities=  np.linspace(1, 100, class_count) # linear distribution
        probabilities = normalize_probability(probabilities) # 
        #if i == 0:
        #    print(probabilities)
        np.random.shuffle(probabilities)
        teachers_probabilities.append(probabilities)
    return np.array(teachers_probabilities)

def generate_uniform_probability_of_drawing(teacher_count, class_count):
    """Create a probability distribution for the teacher to draw from each class in a non-uniform way

    Args:
        teacher_count (_type_): _description_
        class_count (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array([normalize_probability(np.ones(class_count)) for _ in range(teacher_count)])

def normalize_probability(probability_v):
    """Normalize a probability vector

    Args:
        probability_v (_type_): _description_

    Returns:
        _type_: _description_
    """
    return probability_v / np.sum(probability_v)



def teacher_dataset_generation_game(teachers_probabilities, class_distributions):   

    class_count = len(teachers_probabilities[0])
    teacher_count = len(teachers_probabilities)
    teacher_indices = {i : [] for i in range(teacher_count)}

    # randomize class_distributions beforehand
    for i in range(len(teachers_probabilities)):
        np.random.shuffle(teachers_probabilities[i])

    class_distributions = {k: list(v) for k,v in class_distributions.items()}
    
    class_distribution_counts = np.array([len(class_distributions[i]) for i in range(len(class_distributions))])



    index_counts = np.array(teachers_probabilities)
    index_counts = (index_counts / sum(index_counts)) * np.array(class_distribution_counts) # normalize by column (so that each teacher has the same )
    total_sum = sum(sum(    index_counts))

    pts_per_teacher = total_sum // teacher_count
    for i in range(teacher_count): # renormalize rows
        index_counts[i] = np.round(index_counts[i] / sum(index_counts[i]) * pts_per_teacher)
    # index_counts is the number of points each teacher gets to draw from each class


    for teacher in range(teacher_count): # each teacher takes a turn
        for class_ind in range(class_count):
            
            count = int(index_counts[teacher][class_ind]) # the number of points the teacher has to draw from this class
            #print(teacher_indices[teacher])
            #print(type(class_distributions[class_ind]))
            ##print(count)
            #
            #print(type(class_distributions[class_ind][:count]))
            teacher_indices[teacher].extend(class_distributions[class_ind][:count]) # draw the points

            class_distributions[class_ind] = class_distributions[class_ind][count:] # remove the points that were drawn

    return teacher_indices

####

def get_teacher_datasets(dataset, teacher_count, balance_type = "balanced", distribution_type="IID"):
    """ 
    returns a set of datasets, 
        which can be *drawn* from : a balanced or unbalanced distribution
        which can be *sampled* : IID or non-IID across teachers
    
    returns:
        [Subset(dataset, indices) for indices in teacher_indices]
    """
    class_count = len(dataset.classes)
    class_ordering = np.arange(class_count)
    class_to_dataset = get_class_to_dataset(dataset)
    
    # unbalanced Data
    unbalanced_class_distribution_counts = get_zipf_class_distribution_counts_from_dataset(dataset, min_percentage= 40)
    if balance_type == "unbalanced":
        class_distribution_counts = unbalanced_class_distribution_counts
    elif balance_type == "balanced":
        # make sure the balanced dataset doesnt have too many points 
        balanced_class_distribution_counts = [sum(unbalanced_class_distribution_counts)// class_count for _ in range(len(dataset.classes))]
        class_distribution_counts = balanced_class_distribution_counts
    else:
        raise Exception("balance_type must be either balanced or unbalanced")
    

    class_to_indices = get_indices_by_class_size(class_ordering=class_ordering, class_distribution_counts=class_distribution_counts, class_to_dataset=class_to_dataset)
    # Non IID

    if distribution_type == "non-IID":
        probs = generate_non_uniform_probability_of_drawing(teacher_count, class_count)
    elif distribution_type == "IID":
        probs = generate_uniform_probability_of_drawing(teacher_count, class_count)
    else:
        raise Exception("distribution_type must be either IID or non-IID")

    teacher_indices= teacher_dataset_generation_game(probs, class_to_indices)

    return [Subset(dataset, teacher_indices[i]) for i in range(teacher_count)]


###############
##############
##############
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
def teacher_dataset_generation_game_first_try(teachers_probabilities, class_distributions):   

    # randomize class_distributions beforehand
    for i in range(len(teachers_probabilities)):
        np.random.shuffle(teachers_probabilities[i])

    class_distributions = {k: list(v) for k,v in class_distributions.items()}

    class_count = len(teachers_probabilities[0])
    teacher_count = len(teachers_probabilities)
    entirely_used = 0
    teacher_indices = {i : [] for i in range(teacher_count)}

    while entirely_used<class_count:
        for teacher in range(teacher_count): # each teacher takes a turn
            teacher_probability = teachers_probabilities[teacher]
            if sum(teacher_probability) == 0:
                continue

            class_drawn = np.random.choice(class_count, p=teacher_probability)
            index_drawn = class_distributions[class_drawn].pop()
            
            if len(class_distributions[class_drawn])==0:
                entirely_used+=1 
                for i in range(teacher_count): # remove the potential of drawing from this class
                    teachers_probabilities[i][class_drawn] = 0
                    if sum(teachers_probabilities[i]) != 0:
                        
                        teachers_probabilities[i] = normalize_probability(teachers_probabilities[i])

            teacher_indices[teacher].append(index_drawn )
    return teacher_indices
