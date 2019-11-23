import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from BColors import BColors

FILE_PATH = "data/spam.csv"
THRESHOLD = 0.5
EPSILON = 1e-10


def read_file(filename):
    read_data = pd.read_csv(filename, header=None)
    return read_data


def standardize(df_to_std):
    x = df_to_std.iloc[:, 0:57].values
    y = df_to_std.iloc[:, 57]

    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)

    df_x = pd.DataFrame(x_scaled)
    df_scaled = df_x.join(y)

    return df_scaled


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(p, y, w, lambda_reg):
    return -np.average(y * np.log(p + EPSILON) + (1 - y) * np.log(1 - p + EPSILON)) \
           + lambda_reg * np.sum(np.square(w)) / (2 * y.size)


def get_x(df):
    return df.iloc[:, 0:-1].values


def get_y(df):
    return df.iloc[:, -1].values


def predict_probability(x, b, w):
    z = np.dot(x, w) + b
    return sigmoid(z)


def train(df, iterations=500, learning_rate=10, lambda_reg=1, verbose=True):
    x = get_x(df)
    y = get_y(df)

    w_train = np.zeros(x.shape[1])
    b = 0

    for it in range(iterations):
        predictions = predict_probability(x, b, w_train)
        error = predictions - y

        gradient_b = np.average(error)
        gradient_w = np.dot(x.T, error)
        regularization = lambda_reg * w_train

        b -= learning_rate * gradient_b
        w_train -= learning_rate * (gradient_w + regularization) / y.size

        if verbose and it % (iterations/5) == 0:
            print("It. %4d\t|\tLoss: %0.4f" % (it, loss(predictions, y, w_train, lambda_reg)))
    return b, w_train


def predict(x, b, w, threshold=0.5):
    prob = predict_probability(x, b, w)
    return prob >= threshold


def accuracy(df, b, w, verbose=True):
    x = get_x(df)
    y = get_y(df)
    predictions = predict(x, b, w, threshold=THRESHOLD)

    acc = np.average(predictions == y)

    if verbose:
        print("Accuracy: %0.4f\n" % acc)
    return acc


def log_reg_sklearn(df):
    x = get_x(df)
    y = get_y(df)

    model = LogisticRegression(C=1e20)
    model.fit(x, y)

    predictions = model.predict(x)
    acc = np.average(predictions == y)
    print("sklearn: %0.4f" % acc)

    return model.coef_


def get_block_data(df, fold, tot_folds):
    fold_size = math.floor(df.shape[0] / tot_folds)

    start_index = fold_size * fold
    end_index = start_index + fold_size

    df_test = df.loc[start_index:end_index]
    df.drop(df.loc[start_index:end_index].index, inplace=True)

    return df, df_test


def shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)


def cross_validation(df, l_r, l_reg, folds=10, verbose=True, sklearn=True):
    avg_acc = 0
    sk_acc = 0

    for i in range(folds):
        if verbose:
            print(BColors.OK_BLUE + "Fold number " + str(i+1) + BColors.END_C)
        tr_data, test_data = get_block_data(df.copy(), i, folds)
        b, w = train(
            tr_data,
            learning_rate=l_r,
            lambda_reg=l_reg,
            verbose=verbose
        )
        avg_acc += accuracy(test_data, b, w, verbose=verbose)

        if sklearn:
            x = get_x(tr_data)
            y = get_y(tr_data)

            model = LogisticRegression(C=1e20)
            model.fit(x, y)

            predictions = model.predict(get_x(test_data))
            sk_acc += np.average(predictions == get_y(test_data))

    avg_acc /= folds
    sk_acc /= folds

    print(BColors.OK_GREEN + "AVG acc: %0.4f" % avg_acc)
    print("AVG sklearn: %0.4f" % sk_acc)


if __name__ == '__main__':
    df = read_file(FILE_PATH)
    df = standardize(df)

    df.insert(0, 'A', pd.Series(np.ones(df.shape[0]), index=df.index))
    df = shuffle(df)

    cross_val = True

    if cross_val:

        cross_validation(
            df,
            l_r=0.1,
            l_reg=0.01
        )

        # for rate in [0.1, 0.05, 0.01, 0.005, 0.001]:
        #     for reg in [0.1, 0.05, 0.01, 0.005, 0.001]:
        #         print(str(rate), str(reg))
        #         cross_validation(
        #             df_shuffled,
        #             verbose=False,
        #             l_r=rate,
        #             l_reg=reg,
        #             sklearn=False
        #         )
    else:
        bias, weights = train(df)

        accuracy(df, bias, weights)

        w2 = log_reg_sklearn(df)
