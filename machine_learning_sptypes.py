
# coding: utf-8

# In[226]:

import csv
from collections import defaultdict
import re
import itertools
import math
import numpy as np
from sklearn import model_selection, ensemble, preprocessing, tree, metrics, externals
import time
from contextlib import contextmanager
import os


memory = externals.joblib.Memory(cachedir='.cache')

# In[7]:

def iterate_rows():
    filename = 'simbadresult.csv'
    if not os.path.isfile(filename):
        compressed_filename = '{}.xz'.format(filename)
        if os.path.isfile(compressed_filename):
            raise OSError('Found compressed file: {}. Uncompress with unxz'.format(
                compressed_filename))
        else:
            raise OSError('Cannot find {}.'.format(filename))

    with open('simbadresult.csv') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            yield row

# SPTYPE_REGEX = re.compile(r'[OBAFGKM][0-9](I|II|III|IV|V)')
# For now only handle main sequence stars
SPTYPE_REGEX = re.compile(r'(?P<typ>[OBAFGKM])(?P<cls>[0-9](\.[0-9]+)?)V')
SP_TYPE_MAPPING = {value: index for (index, value) in enumerate('OBAFGKM')}


def parse_sptype(sptype):
    match = SPTYPE_REGEX.match(sptype)
    if match:
        return match.group(0)

def sptype_float(sptype):
    match = SPTYPE_REGEX.match(sptype)
    if match:
        int_component = SP_TYPE_MAPPING[match.group('typ')]
        decimal_component = float(match.group('cls')) / 9.
        return int_component + decimal_component

@memory.cache
def build_training_data():

    unique_objects = set()
    all_filters = set()
    unique_sptypes = set()

    for i, row in enumerate(iterate_rows()):
        unique_objects.add(row['main_id'])
        all_filters.add(row['filter'])
        sptype = parse_sptype(row['sp_type'])
        if sptype is None:
            continue
        unique_sptypes.add(sptype)

    all_filters = [item for item in sorted(all_filters) if item.isupper()]
    print("{} unique objects".format(len(unique_objects)))
    print('Including {} filters: {}'.format(len(all_filters), all_filters ))


    # In[148]:

    filter_ordering = ['U', 'u', 'B', 'V', 'g', 'R', 'r', 'I', 'i', 'z', 'J', 'H', 'K']
    all_filters.sort(key=lambda f: filter_ordering.index(f))
    print(all_filters)


    # In[149]:

    valid_colours = []
    for start_band, end_band in itertools.product(all_filters, all_filters):
        if start_band.lower() == end_band.lower():
            continue

        if filter_ordering.index(start_band) >= filter_ordering.index(end_band):
            continue

        valid_colours.append((start_band, end_band))


    # In[150]:

    rows = {}
    for i, row in enumerate(iterate_rows()):
        sp_type = parse_sptype(row['sp_type'])
        if not sp_type:
            continue

        mag_label = row['filter']
        mag_value = float(row['flux'])
        obj_id = row['main_id']

        if obj_id in rows:
            if 'sp_type' not in rows[obj_id]:
                rows[obj_id]['sp_type'] = sp_type
            rows[obj_id][mag_label] = mag_value
        else:
            rows[obj_id] = {'sp_type': sp_type}
            for filt in all_filters:
                rows[obj_id][filt] = float('nan')
            rows[obj_id][mag_label] = mag_value

    rows = list(rows.values())


    # In[151]:

    X, y = [], []
    for row in rows:
        entry = []
        for start_band, end_band in valid_colours:
            colour_value = row[start_band] - row[end_band]
            entry.append(colour_value)
        if all(math.isnan(value) for value in entry):
            continue
        X.append(entry)
        y.append(row['sp_type'])
    X, y = [np.array(data) for data in [X, y]]

    return X, y, ["{}-{}".format(a, b) for (a, b) in valid_colours], list(unique_sptypes)

def sptype_error(a, b):
    a = np.array([sptype_float(val) for val in a])
    b = np.array([sptype_float(val) for val in b])

    return np.sum((a - b) ** 2)

sptype_score = metrics.make_scorer(sptype_error, greater_is_better=False)

@memory.cache
def build_predictor(X, y):
    params = {
        'n_estimators': [10, ],
    }
    clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), params)
    clf.fit(X, y)
    return clf.best_estimator_

def build_colours(mags, all_colours):
    NAN = float('nan')
    out = []
    for colour in all_colours:
        (start_band, end_band) = colour.split('-')
        colour_value = mags.get(start_band, NAN) - mags.get(end_band, NAN)
        out.append(colour_value)
    return np.array(out)


@contextmanager
def timeblock(label):
    start = time.time()
    yield
    end = time.time()
    print('{}: time taken: {:.3e}s'.format(label, end - start))


if __name__ == '__main__':

    X, y, colour_names, labels = build_training_data()

    imp = preprocessing.Imputer()
    imp.fit(X)
    X = imp.transform(X)

    clf = build_predictor(X, y)


    def predict_star(name, mags, expected):
        entry = imp.transform(np.atleast_2d(build_colours(mags, colour_names)))
        prediction = clf.predict(entry).item()
        print('{}, expected: {}, got: {}'.format(name, expected, prediction))
        return prediction

    stars = {
        'HD 209458': {
            'mags': {
                'B': 8.21,
                'V': 7.63,
                'J': 6.591,
                'H': 6.37,
                'K': 6.308,
            },
            'expected': 'G0V',
        },
        'WASP-12': {
            'mags': {
                'B': 12.14,
                'V': 11.57,
                'R': 11.6,
                'J': 10.477,
                'H': 10.228,
                'K': 10.188,
            },
            'expected': 'G0V',
        },
        'GJ 1214': {
            'mags': {
                'B': 16.40,
                'V': 14.67,
                'R': 14.394,
                'I': 11.1,
                'J': 9.750,
                'H': 9.094,
                'K': 8.782,
            },
            'expected': 'M4.5V',
        },
        'Kepler-75': {
            'mags': {
                'B': 16.2,
                'R': 14.9,
                'J': 13.665,
                'H': 13.262,
                'K': 13.118,
            },
            'expected': 'K0V',
        },
        'WASP-69': {
            'mags': {
                'B': 10.93,
                'V': 9.87,
                'J': 8.032,
                'H': 7.543,
                'K': 7.459,
            },
            'expected': 'K5V',
        }
    }

    for star in stars:
        with timeblock(star):
            predict_star(star, stars[star]['mags'], stars[star]['expected'])
