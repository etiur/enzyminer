from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report as class_re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
import numpy as np
import pandas as pd


def split_transform(X, Y, states=20):
    """Given X and Y returns a split and scaled version of them"""
    scaling = MinMaxScaler()
    esterase = ['EH51(22)', 'EH75(16)', 'EH46(23)', 'EH98(11)', 'EH49(23)']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=states, stratify=Y)

    X_train = X_train.loc[[x for x in X_train.index if x not in esterase]]
    X_test = X_test.loc[[x for x in X_test.index if x not in esterase]]
    Y_train = Y_train.loc[[x for x in Y_train.index if x not in esterase]]
    Y_test = Y_test.loc[[x for x in Y_test.index if x not in esterase]]

    transformed_x = scaling.fit_transform(X_train)
    transformed_x = pd.DataFrame(transformed_x)
    transformed_x.index = X_train.index
    transformed_x.columns = X_train.columns

    test_x = scaling.transform(X_test)
    test_x = pd.DataFrame(test_x)
    test_x.index = X_test.index
    test_x.columns = X_test.columns

    return transformed_x, test_x, Y_train, Y_test, X_train, X_test


def vote(*args):
    """
    Hard voting for the ensembles

    Parameters
    ___________
    args: list[arrays]
        A list of prediction arrays
    """
    vote_ = []
    index = []
    if len(args) == 2:
        mean = np.mean(args, axis=0)
        for s, x in enumerate(mean):
            if x == 1 or x == 0:
                vote_.append(int(x))
            else:
                vote_.append(args[-1][s])
                index.append(s)  # keep the index of non unanimous predictions

    elif len(args) % 2 == 1:
        mean = np.mean(args, axis=0)
        for s, x in enumerate(mean):
            if x == 1 or x == 0:
                vote_.append(int(x))
            elif x > 0.5:
                vote_.append(1)
                index.append(s)
            else:
                vote_.append(0)
                index.append(s)

    else:
        mean = np.mean(args, axis=0)
        for s, x in enumerate(mean):
            if x == 1 or x == 0:
                vote_.append(int(x))
            elif x > 0.5:
                vote_.append(1)
                index.append(s)
            elif x < 0.5:
                vote_.append(0)
                index.append(s)
            else:
                vote_.append(args[-1][s])
                index.append(s)

    return vote_, index


def print_score(Y_test, y_grid, train_predicted, Y_train, test_index=None, train_index=None, mode=None):
    """ The function prints the scores of the models and the prediction performance """
    score_tuple = namedtuple("scores", ["test_confusion", "tr_report", "te_report",
                                        "train_mat", "test_mat", "train_confusion"])

    target_names = ["class 0", "class 1"]

    # looking at the scores of those predicted by al 3 of them
    if mode:
        Y_test = Y_test.iloc[[x for x in range(len(Y_test)) if x not in test_index]]
        Y_train = Y_train.iloc[[x for x in range(len(Y_train)) if x not in train_index]]
        y_grid = [y_grid[x] for x in range(len(y_grid)) if x not in test_index]
        train_predicted = [train_predicted[x] for x in range(len(train_predicted)) if x not in train_index]

    # Training scores
    train_confusion = confusion_matrix(Y_train, train_predicted)
    train_matthews = matthews_corrcoef(Y_train, train_predicted)
    # print(f"Y_train : {Y_train}, predicted: {train_predicted}")
    tr_report = class_re(Y_train, train_predicted, target_names=target_names, output_dict=True)

    # Test metrics
    test_confusion = confusion_matrix(Y_test, y_grid)
    test_matthews = matthews_corrcoef(Y_test, y_grid)
    te_report = class_re(Y_test, y_grid, target_names=target_names, output_dict=True)

    all_scores = score_tuple(*[test_confusion, tr_report, te_report, train_matthews,
                               test_matthews, train_confusion])

    return all_scores


def to_dataframe(score_list, name):
    """ A function that transforms the data into dataframes"""
    matrix = namedtuple("confusion_matrix", ["true_n", "false_p", "false_n", "true_p"])

    # Taking the confusion matrix
    test_confusion = matrix(*score_list.test_confusion.ravel())
    training_confusion = matrix(*score_list.train_confusion.ravel())

    # Separating confusion matrix into individual elements
    test_true_n = test_confusion.true_n
    test_false_p = test_confusion.false_p
    test_false_n = test_confusion.false_n
    test_true_p = test_confusion.true_p

    training_true_n = training_confusion.true_n
    training_false_p = training_confusion.false_p
    training_false_n = training_confusion.false_n
    training_true_p = training_confusion.true_p

    # coonstructing the dataframe
    dataframe = pd.DataFrame([test_true_n, test_true_p, test_false_p, test_false_n, training_true_n,
                              training_true_p, training_false_p, training_false_n, score_list.test_mat,
                              score_list.train_mat])

    dataframe = dataframe.transpose()

    dataframe.columns = ["test_tn", "test_tp", "test_fp", "test_fn", "train_tn", "train_tp",
                         "train_fp", "train_fn", "test_Mat", "train_Mat", ]
    dataframe.index = name

    te_report = pd.DataFrame(score_list.te_report).transpose()
    tr_report = pd.DataFrame(score_list.tr_report).transpose()
    te_report.columns = [f"{x}_{''.join(name)}" for x in te_report.columns]
    tr_report.columns = [f"{x}_{''.join(name)}" for x in tr_report.columns]

    return dataframe, te_report, tr_report
