import pandas as pd
from sklearn import preprocessing
import numpy as np
from multiprocessing import Pool
import sys
# Local module import
from .features import SampEn, AUC, autocorr, average_PSD

# Function to trace feature progress.


def percentage_coroutine(to_process, print_on_percent=0.05):
    print("\nStarting progress percentage monitor \n")

    processed = 0
    count = 0
    print_count = to_process * print_on_percent
    while True:
        yield
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed) / float(to_process))
            print("%d%% finished  \r" % pct)


def trace_progress(func, progress=None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf


def parallel_apply(f, X, n_workers=1):
    """
    Parallelised application of function to dataframe.
    Inputs:
    :param f: function to apply
    :param X: dataframe to apply function to.
    :param n_workers: Number of processors to use. (Default: 1)
    """
    results = []
    num_tasks = len(X)
    with Pool(processes=n_workers) as p:
        for i, result in enumerate(p.imap(f, X.as_matrix(), 1)):
            sys.stderr.write('\rdone {0:.0%}\r'.format(i / num_tasks))
            results.append(result)

    return results


def feature_engineering(X, y, groups, n_workers=1):
    n_workers = n_workers
    features = pd.DataFrame()
    features['Artefact'] = y.values
    features['Subject'] = groups.values
    print('Calculating AUC')
    features['AUC'] = parallel_apply(AUC, X, n_workers)
    #min_max_scaler = preprocessing.MinMaxScaler()
    # auc = X.apply(trace_progress(
    #    np.trapz, progress=co2), raw=True, axis=1)

    #features['AUC'] = min_max_scaler.fit_transform(auc)
    print('Calculating PSD')
    psd = parallel_apply(average_PSD, X, n_workers)
    # features['PSD'] = X.apply(trace_progress(
    #    average_PSD, progress=co2), raw=True, axis=1)
    features['PSD'] = psd

    print('Calculating AutoCorr')
    ac = parallel_apply(autocorr, X, n_workers)
    features['AutoCorr'] = ac
    # features['AutoCorr'] = X.apply(trace_progress(
    #    lambda x: x.autocorr(lag=1), progress=co2), axis=1)

    print('Calculating SampEn')
    sampen = parallel_apply(SampEn, X, n_workers)
    # features['SampEn'] = X.apply(trace_progress(
    #    SampEn, progress=co2), raw=True, axis=1)
    features['SampEn'] = sampen
    features.index = X.index
    return features
