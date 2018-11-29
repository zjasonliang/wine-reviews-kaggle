import json
import os
import pickle


def dump_data():
    pickle_filename = 'winemag-data-130k-v2.pickle'

    if not os.path.isfile(pickle_filename):
        filename = 'winemag-data-130k-v2.json'
        with open(filename, 'r') as f:
            data = json.load(f)

        import random
        random.shuffle(data)

        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)


def get_data():
    dump_data()

    pickle_filename = 'winemag-data-130k-v2.pickle'

    with open(pickle_filename, 'rb') as f:
        return pickle.load(f)


def train_valid_test_split(data):
    # _train = data[:len(data)//3]
    # _valid = data[len(data)//3: len(data)//3 * 2]
    # _test = data[len(data)//3 * 2:]
    _train = data[:len(data) // 10 * 9]
    _valid = data[len(data) // 10 * 9:]
    _test = []
    return _train, _valid, _test


def get_train_valid_test_data():
    return train_valid_test_split(get_data())


def mse(y, y_hat):
    assert len(y) == len(y_hat)
    return sum([(p[0] - p[1]) ** 2 for p in zip(y, y_hat)]) / len(y)
