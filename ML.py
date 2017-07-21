import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import abroad.machine_learning as ML

# Import machine learning code
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.exceptions import UndefinedMetricWarning
import os, time, pprint, warnings
warnings.simplefilter('ignore', UndefinedMetricWarning)
warnings.simplefilter('ignore', UserWarning)
import datetime


today=datetime.datetime.now().strftime('%d%b%yT%H%M')
DATAPATH = os.path.join('.', 'data', today)
classification_path = os.path.join(DATAPATH,"classification_scores")
training_path = os.path.join(DATAPATH,"partitioned_data", 'training')
testing_path = os.path.join(DATAPATH,"partitioned_data", 'testing')
prediction_path=os.path.join(DATAPATH,"predictions")
os.makedirs(training_path)
os.makedirs(testing_path)
os.makedirs(prediction_path)
os.makedirs(classification_path)
classification_file=os.path.join(classification_path,"all-classification-reports.txt")



def generate_features():

    df7_dir = './data/df_7/'
    df13_dir = './data/df_13/'

    targets = {"All": ['Control', 'Horizontal', 'Vertical', 'Pressure', 'Frown', 'Ambient Light', 'Torch Light'],
              "Light":['Control', 'Ambient Light', 'Torch Light'],
          "Motion":['Control', 'Horizontal', 'Vertical', 'Pressure', 'Frown']}

    artefact_key = {"All": " ",
                   "Light": " light ",
                   "Motion": " motion "}

    features = [{"data":pd.read_csv(os.path.join(df7_dir,'parallel_features_7_%s.csv'%(x)), index_col=0),
                 "sensor": 7, "targets":targets[x],
                 "artefact": artefact_key[x],
                 "type": x} for x  in ['All', 'Light', 'Motion']]
    features.extend([{"data":pd.read_csv(os.path.join(df13_dir,'parallel_features_13_%s.csv'%(x)), index_col=0),
                      "sensor": 13, "targets":targets[x],
                      "artefact": artefact_key[x],
                      "type": x} for x  in ['All', 'Light', 'Motion']])

    return features


def classification(train, test, data_name=""):

    # Split data and then binarise y-values
    rand_state = np.random.randint(100)

    splitter = list(
        GroupShuffleSplit(n_splits=10, random_state=rand_state).split(train['X'], train['y'], train['groups']))


    y_train = train['y']
    y_test = test['y']



    rfc_param_grid = [
        {"classify__max_depth": [3, None],
         "classify__max_features": [1, 2, 3, 4],
         "classify__min_samples_split": [1, 5, 10],
         "classify__min_samples_leaf": [2, 5, 10],
         "classify__bootstrap": [True, False],
         "classify__criterion": ["gini", "entropy"]}
    ]


    rfc = RandomForestClassifier(n_estimators=1000, class_weight="balanced")


    f1_score = metrics.make_scorer(metrics.f1_score, average="weighted")



    robust_rfc = Pipeline([
        ('normalise', preprocessing.RobustScaler()),
        ('classify', rfc)
    ])



    classifier_info = {"params":None, "preds": None, "clf": None}

    now = time.time()
    grid = GridSearchCV(robust_rfc, rfc_param_grid, scoring=f1_score, cv=splitter, n_jobs=-2, iid=False)
    grid.fit(train['X'], y_train)
    tt = time.time() - now
    tt_format = "{0} hours, {1} mins, {2} seconds and {3} ms\n".format(tt // 3600, tt % 3600 // 60, tt % 60 // 1,
                                                                       tt // 1)
    print("Training for \"{0}\" took {1}".format(data_name, tt_format))

    rfc_file_output = """Using {0} data\n\nUsing {1} classifier.
====================\n
Best parameters set found on development set:\n\n{2}\n
Detailed Classification Report
==============================\n
The model is trained on the full training set.
The scores are computed on the full test set.\n
{3}\n"""

    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    grid_scores = "\n".join(["%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params) for mean, std, params in zip(means, stds, grid.cv_results_['params'])])
    with open(os.path.join("logs","grid_scores.log"), "a") as gsf:
        gsf.write("Scores for {}".format(data_name))
        gsf.write(grid_scores)

    classifier_info['params'] = grid.best_params_
    y_true, y_pred = y_test, grid.predict(test['X'])
    classifier_info['preds'] = y_pred
    classifier_info['clf'] = grid.best_estimator_
    clf_report = metrics.classification_report(y_true, y_pred)
    best_param_string = "\n".join(['\t%s: %r' % (param_name, classifier_info['params'][param_name]) for param_name in sorted(classifier_info['params'])])

    print(rfc_file_output.format(data_name, "rfc", best_param_string, clf_report))
    prediction_file = os.path.join(prediction_path, "predictions-{}.txt".format(data_name.replace(" ", "")))
    report_file = os.path.join(prediction_path, "clf-report-{}.txt".format(data_name.replace(" ", "")))

    with open(classification_file, "a") as cf:
        cf.write(rfc_file_output.format(data_name, "rfc", best_param_string, clf_report))

    with open(prediction_file, "w") as pf:
        for item in classifier_info['preds']:
            pf.write("%s\n" % item)

    with open(report_file, "w") as rf:
        rf.write(clf_report)



    return classifier_info


def pipeline(split_data, data_name=""):
    now_pipeline = time.time()
    info = classification(split_data[0], split_data[1], data_name)
    tt = time.time()-now_pipeline
    tt_format = "{0} hours, {1} mins, {2} seconds and {3} ms\n".format(tt//3600, tt%3600//60, tt%60//1, tt//1)
    print("Training and testing for \"{0}\" took {1}".format(data_name, tt_format))
    return info

if __name__=="__main__":

    feature_sets= ['Sensor 7 - All', 'Sensor 7 - Light', 'Sensor 7 - Motion', 'Sensor 13 - All', 'Sensor 13 - Light', 'Sensor 13 - Motion']

    features = generate_features()

    for idx, df in enumerate(features):
        print("Processing {type} artefacts for sensor {sensor}".format(**df))
        split_data = ML.test_train_split(df['data'], seed=12)
        print("Training on subjects {}\n".format(set(split_data[0]['groups'])))
        print("Testing on subjects {}\n".format(set(split_data[1]['groups'])))
        fname = "s{sensor}-{type}".format(**df)
        # Write data to file for later use
        for k, v in split_data[0].items():
            v.to_csv(os.path.join(training_path,"%s-%s.csv"%(fname,k)),index=False)
        for k, v in split_data[1].items():
            v.to_csv(os.path.join(testing_path,"%s-%s.csv"%(fname,k)),index=False)
        print("Testing for %s" % feature_sets[idx])
        info = pipeline(split_data, data_name = feature_sets[idx])
        pprint.pprint(info, depth=2)





