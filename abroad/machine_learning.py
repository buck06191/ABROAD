import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import cycle
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import svm

import scipy

from .feature_engineering import feature_engineering


def data_separation(df):
    """
    Function to split a dataframe up into X, y and groups
    :param df: Dataframe to split. Must have Artefacts and Subject columns
    :return: (X, y, groups)
    """
    X = df.drop(['Artefact', 'Subject'], axis=1)
    y = df['Artefact']
    groups = df['Subject']
    return(X, y, groups)


def feature_creation(df, n_workers=1):
    X, y, groups = data_separation(df)

    features = feature_engineering(X, y, groups, n_workers)

    return features


def test_train_split(df):
    """
    Function to split a dataframe of engineered features by subject and output
    required Dataframes.
    :param df: Input dataframe from csv file.
    :return: list of dicts of form [[train_data],[test_data]], with data being
    X,  y, groups.
    """
    rand_state = np.random.randint(100)
    X, y, groups = data_separation(df)
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=10, random_state=rand_state).split(X, y, groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_true = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    return [{'X': X_train, 'y': y_train, 'groups': groups_train},
            {'X': X_test, 'y': y_true, 'groups': groups_test}]


def motion_light_split(features, artefact=None):
    if type(artefact) == str:
        artefact = artefact.lower()
    assert artefact in ['light', 'motion',
                        None], 'Please provide a valid artefact type'
    if artefact == 'light':
        new_features = features.drop(
            features[features['Artefact'].isin({1, 2, 3, 4})].index, axis=0)
    elif artefact == 'motion':
        new_features = features.drop(
            features[features['Artefact'].isin({5, 6})].index, axis=0)
    else:
        print("No split required. \n")
        new_features = features

    return new_features


def roc_area(clf, X_test, y_test, n_classes):
    """
    Method of averaging the AUROC scores for each class
    :param clf: fitted classifier in question
    :param X_test: test input data
    :return: ROCAUC
    """
    y_score = clf.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def classification(train):
    """
    Function to classify the training data output from train_test_split
    :param train: train output from test_train_split function
    :return: False positive rate, true positive rate and AUROC
    """

    rand_state = np.random.randint(100)

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=10,
                                                 random_state=rand_state).
                               split(train['X'], train['y'], train['groups']))

    # Binarize the training data
    y = preprocessing.label_binarize(train['y'], classes=list(set(train['y'])))
    n_classes = y.shape[1]

    classifier = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear',
                                                        probability=True,
                                                        random_state=rand_state))

    classifier.fit(train['X'].iloc[train_idx], y[train_idx])

    fpr, tpr, auroc = roc_area(classifier, train['X'].iloc[
                               test_idx], y[test_idx], n_classes)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auroc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, auroc, n_classes, classifier


def final_test(test, classifier):
    """
    Function to classify the test data output from train_test_split
    :param test: test output from train_test_split
    :param classifier: classifier from the classification method
    :return: False positive rate, true positive rate and AUROC
    """

    y = preprocessing.label_binarize(test['y'], classes=list(set(test['y'])))
    n_classes = y.shape[1]
    fpr, tpr, auroc = roc_area(classifier, test['X'], y, n_classes)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auroc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, auroc, n_classes


def ROC_plot(fpr, tpr, auroc, n_classes, datetime,
             target_names, artefact, sensor_num, split_type):
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(auroc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(auroc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(sns.color_palette("husl", n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(target_names[i], auroc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC scores for all%sartefacts - Sensor %s (%s)' %
              (artefact, sensor_num, split_type))
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("figures/ROC_Curve_%s_%s" % (datetime, split_type),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
