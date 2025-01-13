# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:51:00 2024

@author: Anusk
"""

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

path = "./"
filename_read = os.path.join(path,"zoo.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])
df = df.drop(columns=['animal_name'])

tip=encode_text_index(df,"class_type")
#X,y = to_xy(df,"class_type")
X = df.drop(columns=['class_type']).values.astype(np.float32)
y = df['class_type'].values.astype(np.float32)

# Inițializăm KFold 
kf = KFold(n_splits=6)

# Listele pentru stocarea rezultatelor
oos_y = []
oos_pred = []
fold = 0


# Iterăm peste fiecare fold
for train_index, test_index in kf.split(X):
    fold += 1
    print("Fold #{}".format(fold))
        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    np.shape(X_train)
    #crearea arborelui de decizie
    dt = DecisionTreeClassifier(criterion='gini', max_depth=None,min_samples_leaf=1)

    #antrenarea modelului
    dt.fit(X_train, y_train)

    #realizarea predictiei
    y_pred = dt.predict(X_test)
    

    oos_y.append(y_test)
    oos_pred.append(y_pred)
    

    print("Acuratetea de clasificare pentru foldul %d: %.2f" % (fold, accuracy_score(y_test, y_pred)))
#------------------------------------------------------------------------------- 
# Concatenăm toate predicțiile și etichetele reale
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)

# Calculăm acuratetea medie peste toate foldurile
avg_accuracy = accuracy_score(oos_y, oos_pred)
print('Acuratetea medie a modelului Arbori de decizie: {:.2f}'.format(avg_accuracy))
#matricea de confuzie
matrice = confusion_matrix(oos_y, oos_pred)
print("Matricea de confuzie finală:\n", matrice)
#-------------------------------------------------------------------------------
#O varianta de reprezentare grafica a valorilor prezise langa cele reale
def chart_class(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('Clase')
    plt.legend()
    plt.title("Arbori de decizie")
    plt.show()
    

# Plot the chart
chart_class(oos_pred.flatten(),oos_y)
#-------------------------------------------------------------------------------
#arborele
from matplotlib import pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt, feature_names=df.drop(columns=['class_type']).columns.tolist(),class_names=tip.tolist(),
                   filled=True)
#---------------------------------------
