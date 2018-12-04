from utils import *

import matplotlib.pyplot as plt
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report


def linreg_bigram_itfidf(lam=1, min_df=10, tfidf=True):
    data = get_data()
    train, valid, test = train_valid_test_split(data)
    # train, valid, test = get_train_valid_test_data()

    model_path = 'models/'
    model_pickle_filename = 'model2_lam%.2f_min-df%d_tfidf-%s.pickle' % (lam, min_df, tfidf)
    filename = model_path + model_pickle_filename

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model, idf, vocab, feat_scaler = pickle.load(f)
    else:
        X_train = [d['description'] for d in train]
        X_train, idf, vocab = raw_texts_to_vector(X_train, vocabulary=None, min_df=min_df)

        # Append new feature columns to X_train
        X_train, feat_scaler = append_new_features(X_train, train)

        y_train = [float(d['points']) for d in train]

        # Training
        model = Ridge(lam).fit(X_train, y_train)
        # print(model.coef_)
        with open(filename, 'wb') as f:
            pickle.dump((model, idf, vocab, feat_scaler), f)

    # Validate the model
    X_valid = [d['description'] for d in valid]
    X_valid, _, _ = raw_texts_to_vector(X_valid, vocabulary=vocab, use_tfidf=False, min_df=min_df)
    X_valid = apply_idf(X_valid, idf)

    # Append new feature columns to X_train
    X_valid, _ = append_new_features(X_valid, valid, scaler=feat_scaler)

    y_valid = [float(d['points']) for d in valid]

    predictions = model.predict(X_valid)

    print('Lambda =', lam)
    print('MSE on valid:', mse(y_valid, predictions))


if __name__ == '__main__':
    lams = [0, 0.1, 0.5, 1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100]
    for lam in lams:
        linreg_bigram_itfidf(lam=lam, min_df=50)
