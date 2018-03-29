import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import random
import subprocess as sp

random.seed(10)
np.random.seed(10)


def make_png(input_name, output_name):
    png = sp.run(['dot', '-Tpng', input_name, '-o', output_name], stdout=sp.PIPE)
    print(png.stdout.decode('utf-8'))


features = ['age', 'blood', 'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell', 'pus cell clumps',
            'bacteria', 'blood glucose random', 'blood urea', 'serum creatinie', 'sodium', 'potassium', 'hemoglobin',
            'packed cell volume', 'white blood cell count', 'red blood cell count', 'hypertension', 'diabetes mellitus',
            'coronary artery disease', 'appetite', 'pedal edema', 'anemia', 'class', 'trash']

# there is a 26th column due to an extra trailing comma
# the column was named 'trash'

data_original = pd.read_csv('Data\chronic_kidney_disease.arff', header=None, skipinitialspace=True, skiprows=29,
                            sep=',', index_col=False, names=features, na_values=['?', '\t?'])

# CLEANING
data = data_original.copy()
data = data.loc[:, :'class']

# the DecisionTreeRegressor works only with numerical features
# we have to substitute the strings by numbers
# normal/notpresent/no/poor/notckd => 0
# abnormal/present/yes/good/ckd => 1
to_replace = ['yes', 'no', 'present', 'notpresent', 'abnormal', 'normal', 'good', 'poor', '\tyes', '\tno',
              'ckd', 'notckd']
value = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
data = data.replace(to_replace, value)

# First approach: drop NaN
data_drop = data.dropna()

# Decision Tree
clf = DecisionTreeClassifier('entropy')

domain_drop = data_drop.loc[:, :'anemia']
target_drop = data_drop.loc[:, 'class']
target_drop = target_drop.astype('int')
clf_tree_drop = clf.fit(domain_drop, target_drop)
class_names = ['notckd', 'ckd']
dot_data_drop = export_graphviz(clf_tree_drop, out_file="Output\Tree_drop3.dot", feature_names=features[0:24],
                                class_names=True, filled=True, rounded=True, special_characters=True)

make_png('Output\Tree_drop3.dot', 'output\Tree_drop3.png')
