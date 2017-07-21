import pandas as pd
import numpy as np
import abroad.machine_learning as ML
from multiprocessing import Pool, TimeoutError, cpu_count
from datetime import datetime
import os
DATAPATH = os.path.join('.', 'data')



def parallel_split(df, n_workers):
    x = [df.shape[0] // n_workers * k for k in range(n_workers)]
    x.append(df.shape[0])
    return [df[x[k]:x[k + 1]] for k in range(n_workers)]


def output_join(list_of_dataframes):
    return pd.concat(list_of_dataframes).sort_index()


def parallel_batch(df, n_workers=4):
    features = ML.feature_creation(df, n_workers)
    # features = output_join(results)
    return features


def artefact_name(x):
    return {"light": "Light",
            "motion": "Motion",
            None: "All"}.get(x, 'INVALID')


def configure_pipeline(sensor_num, n_workers=4, artefact_type=None,
                       targets=None):
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    config = {"n_workers": n_workers,
              "artefact_type": artefact_type,
              "date": date,
              "sensor_num": sensor_num,
              'target_names': targets}
    return config


def pipeline(df, config):
    artefact_key = {"All" : " ",
               "Light" : " light ",
               "Motion": " motion "}
    features = parallel_batch(ML.motion_light_split(
        df, config['artefact_type']), config['n_workers'])
    artefact = artefact_name(config['artefact_type'])
    features.to_csv(os.path.join(DATAPATH,
                                 'df_%s' % config['sensor_num'],
                                 'parallel_features_%s_%s_%s.csv' %
                                 (config['sensor_num'], artefact, config['date'])))
    #split_data = ML.test_train_split(features)
    #fpr, tpr, auroc, n_classes, clf = ML.classification(split_data[0])
    #ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config[
    #            'target_names'], config['sensor_num'], 'Training')
    #fpr, tpr, auroc, n_classes = ML.final_test(split_data[1], clf)
    #ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config[
    #            'target_names'], config['sensor_num'], 'Test')

    return True


def main_run(df, sensor, n_workers):
    targets = ['Control', 'Horizontal', 'Vertical', 'Pressure',
                         'Frown', 'Ambient Light', 'Torch Light']
    config = configure_pipeline(sensor, n_workers=n_workers, targets=targets)
    pipeline(df, config)

    config = configure_pipeline(sensor, n_workers=n_workers,
                                artefact_type='light',
                                targets=[targets[i] for i in [0, -2, -1]])
    pipeline(df, config)

    config = configure_pipeline(sensor, n_workers=n_workers,
                                artefact_type='motion',
                                targets=targets[:-2])
    pipeline(df, config)

    return True


if __name__ == '__main__':
    df_7 = pd.read_csv('data/raw_sensor_7.csv')
    df_13 = pd.read_csv('data/raw_sensor_13.csv')

    n_workers = cpu_count() - 1
    main_run(df_7, 7, n_workers)
    main_run(df_13, 13, n_workers)
