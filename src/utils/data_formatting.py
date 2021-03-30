import itertools
import pandas as pd
import numpy as np

from typing import Tuple, List, Dict

Dataset = Tuple[np.array, np.array]
AgentDataset = Dict[int, Dataset]
Weights = List[np.array]
GroupIndices = Dict[int,np.array]

"""
    Construct a pandas DataFrame for the training dataset
"""
def construct_df(x_train: np.array, y_train: np.array) -> Tuple[pd.DataFrame, GroupIndices]:
    if x_train.ndim > 2:
        # Reduce the dimensions to 1
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    df = pd.DataFrame(x_train)
    df['label'] = y_train
    # Dictionary mapping label to a list of indices in x_train/y_train
    group_indices = df.groupby(by='label').indices
    return df, group_indices


"""
    Attempt to evenly distribute the number n in a k-dimensional array (to get an IID distribution)
    :returns k-dimensional array such that sum(x)=n
"""
def distribute_number(n: int, k: int) -> np.array:
    div, r = divmod(n, k)
    x = np.full(k, div, 'int')
    if r > 0:
        x[:r] = np.full(r, div + 1)
    return x


"""
    :param samples: 10-dimensional array. Index i stores the number of samples to take from class i
    :param group_indices: Dictionary mapping labels to their corresponding indices in x_train, y_train
    :returns list of tuples that store image and its corresponding label and the updated group_indices
"""
def __partition_data_client(samples: np.array, group_indices: GroupIndices, x_train: np.array, y_train: np.array) -> \
Tuple[List[Dataset], GroupIndices]:
    # The total number of training datapoints
    num_samples = np.sum(samples)
    client_x_train = np.empty((num_samples, 28, 28, 1))
    client_y_train = np.empty(num_samples, 'int')

    indices = []
    for label, label_indices in group_indices.items():
        # How many samples to store for the particular label
        num_samples_label = samples[label]
        # The number of images left to sample from with the label
        num_label = len(label_indices)
        if num_label >= num_samples_label:
            # Uniformly select num_samples_label indices to sample from
            indices_label = np.random.choice(label_indices, num_samples_label, replace=False)
        else:
            # Select all of the remaining data with this particular label
            indices_label = label_indices
        indices.append(indices_label)
    # Flattened list of indices to sample from
    indices = list(itertools.chain(*indices))
    # indices = np.concatenate(indices, axis=None).astype('int')
    # Remove the sampled indices from group_indices => each client has unique training dataset
    group_indices: Dict[int, np.array] = {k: np.array(list(set(v) - set(indices))) for k, v in group_indices.items()}

    # Extract the image and label from the list of indices
    client_x_train, client_y_train = x_train[indices], y_train[indices]
    client_dataset: List[Tuple[np.array, np.array]] = list(zip(client_x_train, client_y_train))

    return client_dataset, group_indices


"""
    Partition the training dataset amongst k clients for each iteration
    The overall training data distribution for each client is approximately IID - NOT for each dataset per round
    :returns List of client dataset - a dictionary mapping iteration [1,iteration] to its dataset
"""
def partition_data_iid(x_train: np.array, y_train: np.array, k: int, iterations: int) -> List[AgentDataset]:
    N = x_train.shape[0]
    # Each entry i stores the number of training images for client i
    data_dist = distribute_number(N, k)
    df, group_indices = construct_df(x_train, y_train)

    client_datasets = []
    for i in range(k):
        d_size = data_dist[i]
        client_i_data_dict = {}
        # Attempt to get a IID feature-distribution for each client i
        client_i_samples = distribute_number(d_size, 10)
        client_i_data, group_indices = __partition_data_client(client_i_samples, group_indices, x_train, y_train)

        # Unzip dataset into list of x_train and y_train
        client_i_x, client_i_y = zip(*client_i_data)
        client_i_x, client_i_y = list(client_i_x), list(client_i_y)

        # Try to split dataset for each iteration
        client_i_x_split, client_i_y_split = np.array_split(client_i_x, iterations), np.array_split(client_i_y,
                                                                                                    iterations)
        # Update the clients training dataset dictionary
        for t in range(1, iterations + 1):
            x, y = client_i_x_split[t - 1], client_i_y_split[t - 1]
            # Tuple of x_train and y_train (numpy arrays)
            client_i_data_dict[t] = (x, y)
        client_datasets.append(client_i_data_dict)
    return client_datasets

"""
    For each client, display information about their dataset for each iteration
"""
def print_client_dataset(client_datasets: List[AgentDataset]) -> None:
    for i in range(len(client_datasets)):
        print('Client {}'.format(i))
        data = client_datasets[i]
        for t, d in data.items():
            x, y = d
            print('Iteration = {}\tNumber of training images = {}'.format(t, x.shape[0]))
            print(client_dataset_summary(d))

"""
    Given a dataset for a particular iteration, compute the count of training images for each label
"""
def client_dataset_summary(client_dataset: Dataset) -> List[Tuple[np.array, np.array]]:
    x_train, y_train = client_dataset
    y_unique, y_counts = np.unique(y_train, return_counts=True)
    # Label paired with its count
    return list(zip(y_unique, y_counts))
