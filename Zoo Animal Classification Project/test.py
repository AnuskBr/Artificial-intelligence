# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:44:28 2024

@author: Anusk
"""
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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

#ANALIZA SET DE DATE 

df.info()

# Descrierea setului de date (numarul de inregistrari,
#media, abaterea standard, valoarea minima, valoarea maxima etc)
df.describe()

# verificare daca exista valori lipsa
df.isnull().sum()

# determinarea valorilor distincte
data = df.drop_duplicates(subset ="class_type",)
data

#grafic 1
sns.countplot(x='class_type', data=df)
plt.show()



# 2.ANTRENARE SI TESTARE
# Inițializăm KFold cu 5 folduri
kf = KFold(n_splits=5)

# Listele pentru stocarea rezultatelor
oos_y = []
oos_pred = []
fold = 0


# Iteram peste fiecare fold
for train, test in kf.split(X):
    fold += 1
    print("Fold #{}".format(fold))
        
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    
    # Inițializăm și antrenăm modelul Perceptron pentru foldul curent
    ppn = Perceptron(max_iter=50, eta0=0.3, random_state=1, early_stopping=True)
    ppn.fit(X_train, y_train)
    
    # Testăm modelul pe setul de test din fold
    y_pred = ppn.predict(X_test)
    
    # Calculăm și colectăm rezultatele
    oos_y.append(y_test)
    oos_pred.append(y_pred)  
    
    #evaluarea modelului 
    #numararea instantelor clasificate gresit
    print('Instante clasificate gresit: %d' % (y_test != y_pred).sum())
    # Evaluăm acuratețea pentru foldul curent
    accuracy = accuracy_score(y_test, y_pred)
    print('Acuratetea pentru foldul {} este: {:.2f}'.format(fold, accuracy))
#------------------------------------------------------------------------------- 
# acuratetea si matricea de confuzie
oos_y=np.concatenate(oos_y)
oos_pred=np.concatenate(oos_pred)

# acuratetea
avg_accuracy = accuracy_score(oos_y, oos_pred)
print('Acuratetea medie a modelului cu K-fold cross-validation este: {:.2f}'.format(avg_accuracy))

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
    plt.ylabel('output')
    plt.legend()
    plt.show()


# Plot the chart
chart_class(oos_pred.flatten(),oos_y)
#-----------------------------------------------------
# acuratețea pe fiecare clasă
class_accuracy = []
for i in range(len(tip)):
    class_accuracy.append(accuracy_score(oos_y[oos_y == i], oos_pred[oos_y == i]))

# Heatmap pentru acuratețe pe clase
plt.figure(figsize=(8, 6))
sns.heatmap(np.array(class_accuracy).reshape(1, -1), annot=True, cmap='Blues', xticklabels=tip, yticklabels=False)
plt.xlabel('Clasa')
plt.ylabel('Acuratete')
plt.title('Acuratete pe Clase')
plt.show()