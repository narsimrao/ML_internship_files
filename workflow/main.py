import pandas as pd

from preprocessing import preprocessing
from model import model
from testing import model_accuracy

import pickle
import matplotlib.pyplot as plt
import os

titanic_data = pd.read_csv('train.csv')

# Preprocessing
X_train, X_test, Y_train, Y_test = preprocessing.preprocessing(titanic_data)

#Fitting the model
LR_model, RF_model, DT_model = model.model_processing(X_train, X_test, Y_train, Y_test)

# Saving the models
filename = 'models/Logistic_regression.sav'
pickle.dump(LR_model, open(filename, 'wb'))

filename = 'models/RandomForest.sav'
pickle.dump(RF_model, open(filename, 'wb'))

filename = 'models/DecisionTree.sav'
pickle.dump(DT_model, open(filename, 'wb'))

# Testing models

models = os.listdir('models/')
score = []
for i in range(0,len(models)):
    score.append(model_accuracy.Model_score(X_test, Y_test, models[i]) * 100)

max_index = score.index(max(score))

print('The model',models[max_index].split('.')[0],'has the highest accuracy with',round(score[max_index],2),'accuracy.')

plt.figure(figsize=(15,5))
plt.bar(models,score)
plt.show()
