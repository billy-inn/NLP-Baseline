import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Iterating over dataset with mini-batches.

    :param data: dataset created with list(zip(X_1, X_2, ..., y))
    :param batch_size: batch size
    :param num_epochs: number of epochs to iterate
    :param shuffle: whether to shuffle the dataset
    :return: iterable objects with batched data
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            np.random.seed(2018)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def create_label_dict(labels):
    """
    Return a dictionary mapping the labels to numbers.

    :param labels: a list of labels
    :return: a dict
    """
    labels = sorted(list(set(labels)))
    mapping = zip(labels, range(len(labels)))
    return dict(mapping)
