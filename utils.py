import json
import os
import pickle
import string
from collections import defaultdict
import random


def dump_data():
    pickle_filename = 'winemag-data-130k-v2.pickle'

    if not os.path.isfile(pickle_filename):
        filename = 'winemag-data-130k-v2.json'
        with open(filename, 'r') as f:
            data = json.load(f)

        random.shuffle(data)

        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)


def dump_data_remove_duplicate_and_na():
    pickle_filename = 'winemag-data_first150k.pickle'

    if not os.path.isfile(pickle_filename):
        import pandas as pd
        init_data = pd.read_csv("winemag-data_first150k.csv")

        # Drop Duplicates
        parsed_data = init_data[init_data.duplicated('description', keep=False)]

        # Drop NaNs
        parsed_data.dropna(subset=['description', 'points'])

        data = parsed_data.to_dict('records')

        import math
        data = [d for d in data if not math.isnan(d['price'])]

        random.shuffle(data)

        if not os.path.isfile(pickle_filename):
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data, f)


def get_data():
    # dump_data()
    dump_data_remove_duplicate_and_na()

    pickle_filename = 'winemag-data_first150k.pickle'

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


def get_idf_by_word(vectorizer, word):
    idx = vectorizer.vocabulary_[word]
    return vectorizer.idf_[idx]


def get_ith_doc_word_indices(csr_matrix, i):
    return csr_matrix.indices[csr_matrix.indptr[i]:csr_matrix.indptr[i + 1]]


def apply_idf(csr_matrix, idf, use_matrix_mult=True):
    assert csr_matrix.shape[1] == len(idf)

    if use_matrix_mult:
        import scipy.sparse as sp
        _idf_diag = sp.diags(idf, offsets=0,
                             format='csr')

        return csr_matrix * _idf_diag
    else:
        n_doc = csr_matrix.shape[0]

        for i in range(n_doc):
            for word_idx in get_ith_doc_word_indices(csr_matrix, i):
                csr_matrix[i, word_idx] = csr_matrix[i, word_idx] * idf[word_idx]

        return csr_matrix


def get_vocabulary(data, vocab_size):
    word_count = defaultdict(int)
    punctuation = set(string.punctuation)

    for d in data:
        text = ''.join([c for c in d['description'].lower() if c not in punctuation])
        for w in text.split():
            word_count[w] += 1

    counts = [(word_count[w], w) for w in word_count]
    counts.sort(reverse=True)

    # A mapping (e.g., a dict) where keys are terms
    # and values are indices in the feature matrix.
    _dict = dict()
    for idx in range(vocab_size):
        _dict[counts[idx][1]] = idx

    return _dict


def raw_texts_to_vector(text_list, vocabulary=None, min_df=50, use_tfidf=True, norm=None, use_bigram=True, dynamic_vocab_size=True):
    punctuation = set(string.punctuation)

    if not dynamic_vocab_size:
        assert vocabulary is not None

    def tokenizer(doc):
        """
        Used to override `token_pattern` step while preserving the preprocessing and n-grams generation steps.
        Only applies if analyzer == 'word'.

        See: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        :param doc: input raw text
        :return: a list of token
        """
        return ''.join([c for c in doc.lower() if c not in punctuation]).split()

    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if use_bigram:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                         min_df=min_df,
                                         # vocabulary=vocabulary,
                                         norm=norm,
                                         smooth_idf=True,
                                         tokenizer=tokenizer
                                         )
            csr_matrix = vectorizer.fit_transform(text_list)
            # print(vectorizer.vocabulary_)
            vocabulary = vectorizer.vocabulary_
            return csr_matrix, vectorizer.idf_, vocabulary
        else:
            vectorizer = TfidfVectorizer(vocabulary=vocabulary,
                                         norm=norm,
                                         smooth_idf=True,
                                         tokenizer=tokenizer
                                         )
            csr_matrix = vectorizer.fit_transform(text_list)
            # print(vectorizer.vocabulary_)
            return csr_matrix, vectorizer.idf_, None
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        if use_bigram:
            vectorizer = CountVectorizer(vocabulary=vocabulary,
                                         tokenizer=tokenizer,
                                         ngram_range=(1, 2),
                                         min_df=min_df,
                                         # token_pattern=r'(?u)\b\w+\b',
                                         )
            csr_matrix = vectorizer.fit_transform(text_list)
            vocabulary = vectorizer.vocabulary_
            return csr_matrix, None, vocabulary
        else:
            vectorizer = CountVectorizer(vocabulary=vocabulary,
                                         tokenizer=tokenizer,
                                         # token_pattern=r'(?u)\b\w+\b',
                                         )
            csr_matrix = vectorizer.fit_transform(text_list)

            return csr_matrix, None, None


def append_new_features(X, data, scaler=None):
    """
    Append new feature columns to the existing feature matrix X.

    New features include:
    - Price
    - Description length
    - Winery

    :param X: Feature matrix, where each row is a feature vector of a data example
    :param data: raw data
    :return: new feature matrix
    """
    from scipy.sparse import hstack
    import numpy
    import math

    new_features = list()
    new_features.append([len(datum['description']) for datum in data])
    new_features.append([datum['price'] for datum in data])

    # Transpose doc_lengths
    new_features = numpy.array(new_features)
    new_features = new_features.transpose()

    X = hstack((X, new_features))

    # Now do feature scaling
    from sklearn import preprocessing

    # For training data
    if scaler is None:
        scaler = preprocessing.MaxAbsScaler()
        X = scaler.fit_transform(X)
        return X, scaler
    # For test data
    else:
        X = scaler.transform(X)
        return X, None

