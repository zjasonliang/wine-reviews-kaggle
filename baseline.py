from utils import *

import pickle
import os
from sklearn.linear_model import Ridge


def baseline(lam=1, vocab_size=20000, tfidf=True):
    """
    Baseline model:
    - Fixed vocabulary
    - Unigram only

    :param lam:
    :param vocab_size:
    :param tfidf:
    :return:
    """

    data = get_data()
    train, valid, test = train_valid_test_split(data)
    # train, valid, test = get_train_valid_test_data()

    vocab = get_vocabulary(train, vocab_size=vocab_size)

    model_path = 'models/'
    model_pickle_filename = 'baseline_lam%.2f_vocab%d_tfidf-%s.pickle' % (lam, vocab_size, tfidf)
    filename = model_path + model_pickle_filename

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model, idf, _ = pickle.load(f)
    else:
        X_train = [d['description'] for d in train]
        X_train, idf, _ = raw_texts_to_vector(X_train, vocabulary=vocab,
                                              use_bigram=False,
                                              dynamic_vocab_size=False)

        y_train = [float(d['points']) for d in train]

        # Training
        model = Ridge(lam).fit(X_train, y_train)
        # print(model.coef_)
        with open(filename, 'wb') as f:
            pickle.dump((model, idf, vocab), f)

    # Validate the model
    X_valid = [d['description'] for d in valid]
    X_valid, _, _ = raw_texts_to_vector(X_valid, vocabulary=vocab,
                                        use_tfidf=False,
                                        use_bigram=False,
                                        dynamic_vocab_size=False)
    X_valid = apply_idf(X_valid, idf)

    y_valid = [float(d['points']) for d in valid]

    predictions = model.predict(X_valid)

    print('Lambda =', lam)
    print('MSE on valid:', mse(y_valid, predictions))


if __name__ == '__main__':
    lams = [0, 0.01, 0.1, 1, 2, 3, 5, 7, 10, 100]
    for lam in lams:
        baseline(lam=lam, vocab_size=10000)
