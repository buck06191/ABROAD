"""
.. module:: merge_data
   :platform: Unix, Windows
   :synopsis: This module contains the :func:`merge_data.df_concat`.

.. moduleauthor:: Joshua Russell-Buckland <joshua.russell-buckland.15@ucl.ac.uk>


"""
import pandas as pd
from . import data_creation as dc
import os
from .. import utility as utility
import copy


BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
assert utility.base_dir_check(BASEDIR, "ABROAD", verbose=True), "incorrect BASEDIR in merge_data"
HDF_FILE = os.path.join(BASEDIR,'data','subject_data.h5')


def df_concat(sensor_num, subject_number):
    """
    Create a pandas dataframe for each subject and then concatenate to generate
    appropriate data structures for use in machine learning.

    :param int sensor_num: The sensor number.

    :param subject_number: either the max subject number or a list of numbers.
    :type subject_number: list or int or float
    """
    df_list = []

    if type(subject_number) == int:
        subject_list = list(range(1, subject_number+1))
    elif type(subject_number) == float:
        subject_list = list(range(1, int(round(subject_number))+1))
    elif type(subject_number) == list:
        subject_list = subject_number
    else:
        print('Not a valid sensor number specification. See help(df_concat)')

    for i in subject_list:
        print('Subject %d' % i)
        data = dc.SubjectData(HDF_FILE, i, sensor_num)
        data.artefact_row_extraction()
        data.group_labels()
        data.label_raw_data()
        data.label_conc_data()
        df_list.append(data.raw_df)
        data.conc_df.to_csv(os.path.join(BASEDIR,'data','conc_subject_%d.csv' % i),
                      index=False)

    all_df = pd.concat(df_list)
    print('\nWriting data to file\n')
    all_df.to_csv(os.path.join(BASEDIR,'data','raw_sensor_%s.csv' % sensor_num),
                  index=False)
    print('\nraw_sensor_%s.csv written to file.\n' % sensor_num)
    return None


def merge_main(sensor_num, hdf_file=HDF_FILE):
    """
    Main function to call in order to merge hdf into csv
    :param sensor_num: Sensor number
    :param hdf_file: hdf file location. (Default: data/subject_data.h5)
    :return: None - will write to file
    """
    df_concat(sensor_num, 8)
    return None
