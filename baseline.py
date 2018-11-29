from utils import *

import matplotlib.pyplot as plt
from collections import defaultdict
import string
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report


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


def raw_texts_to_vector(text_list, verbose=False, vocabulary=None, tfidf=True):
    if tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        text_vector_list = vectorizer.fit_transform(text_list)
        return text_vector_list
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        text_vector_list = vectorizer.fit_transform(text_list)

        if verbose:
            print('Shape of Sparse Matrix: ', text_vector_list.shape)
            print('Amount of Non-Zero occurrences: ', text_vector_list.nnz)

            # Percentage of non-zero values
            density = (100.0 * text_vector_list.nnz / (text_vector_list.shape[0] * text_vector_list.shape[1]))
            print('Density: {}'.format(density))

        return text_vector_list


def baseline(lam=1, vocab_size=20000, tfidf=True):
    data = get_data()

    vocab = get_vocabulary(data, vocab_size=vocab_size)
    train, valid, test = train_valid_test_split(data)

    model_path = 'models/'
    model_pickle_filename = 'lam%.2f_vocab%d_tfidf-%s.pickle' % (lam, vocab_size, tfidf)
    filename = model_path + model_pickle_filename

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    else:
        X_train = [d['description'] for d in train]
        X_train = raw_texts_to_vector(X_train, vocabulary=vocab)

        y_train = [float(d['points']) for d in train]

        # Training
        model = Ridge(lam).fit(X_train, y_train)
        # print(model.coef_)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # Validate the model
    X_valid = [d['description'] for d in valid]
    X_valid = raw_texts_to_vector(X_valid, vocabulary=vocab)

    y_valid = [float(d['points']) for d in valid]

    predictions = model.predict(X_valid)

    print('Lambda =', lam)
    print('MSE on valid:', mse(y_valid, predictions))


if __name__ == '__main__':
    lams = [0, 0.01, 0.1, 1, 2, 3, 5, 7, 10, 100]
    for lam in lams:
        baseline(lam=lam, vocab_size=10000)
