# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:45:55 2018

@author: Henrique
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_original = pd.read_csv("arrhythmia.data", header=None, na_values="?")

data = data_original.dropna(axis=1)
data = data.loc[:,(data != 0).any(axis=0)]
data.loc[data[279]>1,279] = 2

class_id = data[279]
y = data.loc[:,:278]

y1 = y.loc[class_id==1]
y2 = y.loc[class_id==2]

x1 = y1.mean(axis=0)
x2 = y2.mean(axis=0)

dif1 = y.apply(lambda x: x-x1, axis=1)
dif2 = y.apply(lambda x: x-x2, axis=1)
dist1 = dif1.apply(lambda x: x*x, axis=1).sum(axis=1)
dist2 = dif2.apply(lambda x: x*x, axis=1).sum(axis=1)

distance = dist1 - dist2

est_class_id = pd.Series(distance).to_frame()
mask = est_class_id[0] > 0
est_class_id.loc[mask, 0] = 2
est_class_id.loc[~mask, 0] = 1

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for i in range(len(class_id)):
    if class_id[i] == 1:
        if est_class_id.at[i,0] == 1:
            true_negative += 1
        else:
            false_negative += 1
    else:
        if est_class_id.at[i,0] == 2:
            true_positive += 1
        else:
            false_positive += 1

