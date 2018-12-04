from utils import *

import pickle
import os
from sklearn.linear_model import Ridge


def baseline(lam=1, vocab_size=20000, use_tfidf=True):
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
    model_pickle_filename = 'baseline_lam%.2f_vocab%d_tfidf-%s.pickle' % (lam, vocab_size, use_tfidf)
    filename = model_path + model_pickle_filename

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model, idf, _ = pickle.load(f)
    else:
        X_train = [d['description'] for d in train]
        X_train, idf, _ = raw_texts_to_vector(X_train, vocabulary=vocab,
                                              use_tfidf=use_tfidf,
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
                                        use_tfidf=False,  # Will apply IDF later
                                        use_bigram=False,
                                        dynamic_vocab_size=False)
    if use_tfidf:
        X_valid = apply_idf(X_valid, idf)

    y_valid = [float(d['points']) for d in valid]

    predictions = model.predict(X_valid)

    print('Lambda =', lam)

    _mse = mse(y_valid, predictions)
    print('MSE on valid:', _mse)
    return _mse


if __name__ == '__main__':
    vocab_sizes = [1000, 2000, 3000, 4000, 5000,
                   6000, 7000, 8000, 9000, 10000,
                   11000, 12000, 13000, 14000, 15000,
                   16000, 17000, 18000, 19000, 20000]

    mses = []

    for vocab_size in vocab_sizes:
        ret = baseline(lam=0, vocab_size=vocab_size, use_tfidf=False)
        mses.append(ret)

    import matplotlib.pyplot as plt

    plt.plot(vocab_sizes, mses)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('MSE')
    plt.savefig('vocab_size_mse.png')
