# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:48:04 2023

@author: Anusk
"""

import pandas as pd
import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

path = "./"
filename_read = os.path.join(path,"ParisHousing.csv")
filename_write=os.path.join(path,"Housing.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

# Shuffle
np.random.seed(42)
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)
#--------------------------------------------------------------------------
# Preprocess
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)
#--------------------------------------------------------------------------

    
df.drop('numPrevOwners',axis=1,inplace=True)
missing_median(df, 'numberOfRooms')


dataset=df.values
x=dataset[:,0:14] 
y=dataset[:,15]

# Cross-Validate
kf = KFold(2)
    
oos_y = []
oos_pred = []
fold = 0

for train, test in kf.split(x):
    fold+=1
    print("Fold #{}".format(fold))
        
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]
    
    model = Sequential()
    model.add(Dense(700, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(220, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=200)
    
    pred = model.predict(x_test)
    
    oos_y.append(y_test)
    oos_pred.append(pred)        

    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("Fold score (RMSE): {}".format(score))


# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
#-----------------------------------------------------------------
#grafic
plt.figure(figsize=(8, 6))

# Sortează valorile așteptate și prezise
sorted_indices_expected = np.argsort(oos_y)
sorted_indices_predicted = np.argsort(oos_pred)
sorted_expected = oos_y[sorted_indices_expected]
sorted_predicted = np.squeeze(oos_pred[sorted_indices_predicted]) 

# Plotare linie pentru datele așteptate și datele prezise
plt.plot(sorted_expected, label='Expected', color='blue')
plt.plot(sorted_predicted, label='Predicted', color='orange')

plt.ylabel('Output')
plt.title('Expected vs Predicted')
plt.legend()
plt.show()
#-----------------------------------------------------------------

score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
print("Final, out of sample score (RMSE): {}".format(score))
# Write the cross-validated prediction
oos_y = pd.DataFrame(oos_y)
oos_pred = pd.DataFrame(oos_pred)
oosDF = pd.concat( [df, oos_y, oos_pred],axis=1 )
oosDF.to_csv(filename_write,index=False)

