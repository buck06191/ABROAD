import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import cycle
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

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


def score_classification(train, test):
    """
    Function to classify the training data output from train_test_split
    :param train: train output from test_train_split function
    :return: False positive rate, true positive rate and AUROC
    """

    rand_state = np.random.randint(100)

    splitter = list(GroupShuffleSplit(n_splits=10, random_state=rand_state).split(train['X'], train['y'], train['groups']))
    n_groups = len(set(train['groups']))

    lb_train = preprocessing.LabelBinarizer()
    lb_train.fit(train['y'])
    y_train = lb_train.transform(train['y'])

    # Binarize the training data
    lb_test = preprocessing.LabelBinarizer()
    lb_test.fit(test['y'])
    y_test = lb_test.transform(test['y'])

    svc_param_grid = [
        {'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['linear']},
        {'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [0.01, 0.001, 0.0001, 0.00001], 'estimator__kernel': ['rbf']},
    ]

    rfc_param_grid = [
        {"max_depth": [3, None],
         "max_features":  [1, np.ceil(n_groups/2), n_groups],
         "min_samples_split": [1, 3, 10],
         "min_samples_leaf": [1, 3, 10],
         "bootstrap": [True, False],
         "criterion": ["gini", "entropy"]}
    ]


    scores = ['precision_macro', 'recall_macro']
    svr = multiclass.OneVsRestClassifier(svm.SVC(class_weight="balanced", decision_function_shape="ovr"))
    rfc = RandomForestClassifier(n_estimators=100)
    c_dict = {"svr": (svr, svc_param_grid), "rfc": (rfc, rfc_param_grid)}

    classifier_info = {k: {"params":None, "preds": None, "clf": None} for k in scores}
    for score in scores:
        for c in c_dict:
            print("# Now training for score: \t %s using %s" % (score, c))
            clf = GridSearchCV(c_dict[c][0], c_dict[c][1], scoring=score, cv=splitter)

            clf.fit(train['X'], y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            classifier_info[score]['params'] = clf.best_params_
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full training set.")
            print("The scores are computed on the full test set.")
            print()
            y_true, y_pred = y_test, clf.predict(test['X'])
            classifier_info[score]['preds'] = y_pred
            classifier_info[score]['clf'] = clf.best_estimator_
            print(metrics.classification_report(y_true, y_pred))
            print()


    """
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
    """
    return classifier_info


def final_test(test, classifier_info):
    """
    Function to classify the test data output from train_test_split
    :param test: test output from train_test_split
    :param classifier: classifier from the classification method
    :return: False positive rate, true positive rate and AUROC
    """

    lb = preprocessing.LabelBinarizer()
    lb.fit(test['y'])
    y = lb.transform(test['y'])
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
