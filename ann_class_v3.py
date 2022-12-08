#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as kr
import tensorflow.keras.layers as ly
import matplotlib.pyplot as plt


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

classifier = kr.Sequential()
classifier.add(ly.Dense(units=10, input_shape=(8,), kernel_initializer='uniform', activation='sigmoid'))
classifier.add(ly.Dense(units=10, kernel_initializer='uniform', activation='relu'))
classifier.add(ly.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    ###########################################################
    #####                   TRAIN MODEL                   #####
    ###########################################################

pulsar_ANN_Model=classifier.fit(X_train,y_train, batch_size=5 , epochs=50, verbose=1, validation_data=(X_val,y_val))


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
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks(), rotation = 45)

plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.show()

#%%
def get_acc(X,y):
    Predictions=classifier.predict(X)
    
    # Scaling the test data back to original scale
    Test_Data=PredictorScalerFit.inverse_transform(X)
    
    # Generating a data frame for analyzing the test data
    TestingData=pd.DataFrame(data=Test_Data, columns=features)
    TestingData['target_class']=y
    TestingData['PredictedPulsarProb']=Predictions
    
    # Defining the probability threshold
    def probThreshold(inpProb):
        if inpProb > 0.5:
            return(1)
        else:
            return(0)
    
    # Generating predictions on the testing data by applying probability threshold
    TestingData['PredictedPulsar']=TestingData['PredictedPulsarProb'].apply(probThreshold)
    print(TestingData.head())
    
    ###############################################
    from sklearn import metrics
    print('\n######### Testing Accuracy Results #########')
    print(metrics.classification_report(TestingData['target_class'], TestingData['PredictedPulsar']))
    return metrics.confusion_matrix(TestingData['target_class'], TestingData['PredictedPulsar'])

def confusion_matrix(TP,FP,TN,FN):

    col = ["Positive", "Negative"]
    ind = ["Positive", "Negative"]

    matrix=np.array([[TP,FN],[FP,TN]])
    df = pd.DataFrame(matrix,columns=col,index=ind)
    fig = sn.heatmap(df,annot=True,cbar=True,fmt='g',cmap="flare")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix")
    plt.show()
# %%
conf = get_acc(X_train, y_train)
conf2 = get_acc(X_val, y_val)
# %%
import seaborn as sn
confusion_matrix(conf[-1,-1], conf[0,-1], conf[0,0], conf[-1,0])
confusion_matrix(conf2[-1,-1], conf2[0,-1], conf2[0,0], conf2[-1,0])
# %%
