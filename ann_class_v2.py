#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as kr
import tensorflow.keras.layers as ly
import matplotlib.pyplot as plt
import json



    ###########################################################
    #####                   INIT DATASET                  #####
    ###########################################################


df=pd.read_csv('C:\\Users\\Public\\Documents\\IPSA\\SEOULTECH\\COURSES\\MACHINE LEARNING\\datasets\\pulsar_data_train.csv')
df.head()
df = df.dropna().reset_index(drop=True) # remove rows with a Nan value from dataset and reset the index
nb_row = df['target_class'].sum()
df_0 = df.loc[df['target_class'] == 0].sample(n=nb_row,random_state=42)
df_1 = df.loc[df['target_class'] == 1]
df = pd.concat([df_0, df_1])
df = df.sample(frac=1).reset_index(drop=True) # shuffle the rows 

target_var=['target_class']
features=[' Mean of the integrated profile', ' Standard deviation of the integrated profile',
            ' Excess kurtosis of the integrated profile', ' Skewness of the integrated profile',
            ' Mean of the DM-SNR curve', ' Standard deviation of the DM-SNR curve',
            ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']
 
X=df[features].values
y=df[target_var].values


### Sandardization of data ###
### We does not standardize the Target variable for classification
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)

# Split the data into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
    ###########################################################
    #####                   INIT NETWORK                  #####
    ###########################################################

# Function to generate Deep ANN model 
def ann(optimizer, nb_units,a1,a2,a3):


    # Creating the classifier ANN model
    classifier = kr.Sequential()
    classifier.add(ly.Dense(units=nb_units, input_shape=(8,), kernel_initializer='uniform', activation=a1))
    classifier.add(ly.Dense(units=nb_units, kernel_initializer='uniform', activation=a2))
    classifier.add(ly.Dense(units=1, kernel_initializer='uniform', activation=a3))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            
    return classifier


    ###########################################################
    #####               BEST HYPERPARAMETERS              #####
    ###########################################################
#%%
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


hyperparameters={'batch_size':[5,10],
                      'epochs':[30,50],
                    'optimizer':['adam'],
                  'nb_units': [5,10],
                  'a1': ['sigmoid','relu'],
                  'a2': ['sigmoid','relu'],
                  'a3': ['sigmoid'],
                 }


# Creating the classifier ANN
classifierModel=KerasClassifier(ann, verbose=0)

########################################

# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search=GridSearchCV(estimator=classifierModel, param_grid=hyperparameters, scoring='f1', cv=5)

########################################

# Measuring how much time it took to find the best params
import time
StartTime=time.time()

# Running Grid Search for different paramenters
grid_search.fit(X_train,y_train, verbose=1, validation_data=(X_val,y_val))

EndTime=time.time()
print(f'############### Total Time : {round((EndTime-StartTime)/60)} Minutes #############')

# get the results for each combination
dict_results = grid_search.cv_results_
best_params = grid_search.best_params_
list_params = grid_search.cv_results_['params']

"""
# Save the resulting dictionnary because the computations are long
np.save('grid_search_results.npy', dict_results)
np.save('grid_search_best_params.npy', best_params)
np.save('grid_search_list_params.npy', list_params)
"""


#%%
for i in range(len(dict_results['params'])):
  print(f"params {i} :  {dict_results['params'][i]}")

plt.plot([f'params {i+1}' for i in range(len(dict_results['params']))],dict_results['mean_test_score'], marker='*')
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks(), rotation = 45)
plt.title('Parameters score')


########################################
#%%
# printing the best parameters
print('\n                    #### Best hyperparamters ####')
print(best_params)


#%%
best_params = {'a1': 'sigmoid', 'a2': 'relu', 'a3': 'sigmoid', 'batch_size': 5, 'epochs': 50, 'nb_units': 10, 'optimizer': 'adam'}
classifier = ann(best_params['optimizer'],best_params['nb_units'],best_params['a1'],best_params['a2'],best_params['a3'])
pulsar_ANN_Model=classifier.fit(X_train,y_train, batch_size=best_params['batch_size'], 
                                epochs=best_params['epochs'], verbose=1, validation_data=(X_val,y_val))


    ###########################################################
    #####                  PLOT RESULTS                   #####
    ###########################################################

plt.figure()
plt.plot(pulsar_ANN_Model.history['accuracy'], 'dodgerblue')
plt.plot(pulsar_ANN_Model.history['val_accuracy'], 'darkorange')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.show()

plt.figure()
plt.plot(pulsar_ANN_Model.history['loss'])
plt.plot(pulsar_ANN_Model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.show()

#%%
