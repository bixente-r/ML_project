import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize,fmin_tnc
import seaborn as sn
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
import statistics as st
from time import sleep
from tqdm import tqdm

"""
Version 3 projet ML (overall accuracy replaced by preision on 1 and 0) - training on balanced file 50/50 
"""


def convert_X(X):
    for i in X[0:0]:
        dx = {}
        k = 0
        if isinstance((X[i][0]), str) == True:
            for j in range(X.shape[0]):
                if X[i][j] not in dx.keys():
                    dx[X[i][j]] = k
                    k += 1
            for l in range(X.shape[0]):
                for j in dx.keys():
                    if X[i][l] == j:
                        X[i][l] = dx[j]


def convert_y(y):
    dy = {}
    k = 0

    if isinstance((y[0]), str) == True:
        for i in range(0,y.shape[0]):
            if y[i] not in dy.keys():
                dy[y[i]] = k
                k += 1

        for i in range(0,y.shape[0]):
            for j in dy.keys():
                if y[i] == j:
                    y[i] = dy[j]

def init_frame(path, id_col=0):

    df = pd.read_csv(path) # import the data
    df.head()
    df = df.dropna().reset_index(drop=True) # remove rows with a Nan value from dataset and reset the index
    
    nb_row = df['target_class'].sum()
    # print(df)
    df_0 = df.loc[df['target_class'] == 0].sample(n=nb_row,random_state=42)
    df_1 = df.loc[df['target_class'] == 1]
    df = pd.concat([df_0, df_1])
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the rows 

    X = df.iloc[:,id_col:-1] # Separate input from output 
    y = df.iloc[:,-1]  # Check rows and columns from .csv
                        # parameters might change


    corr_matrix = df.corr()

    convert_X(X)
    convert_y(y)
    # print(X)
    # print(y)
    X = np.c_[np.ones((X.shape[0], 1)), X] # add a bias column
    X = np.array([[float(i) for i in e] for e in X]) # convert to float (in case it's not)
    row_nb = X.shape[0]
    traning_nb = round(0.8*row_nb)
    X1 = X[:traning_nb] # training set
    X2 = X[traning_nb:] # validation set


    y = y.to_numpy() # convert to numpy type
    y = y.reshape(len(y),1) # convert to matrix
    y = np.array([[float(i) for i in e] for e in y]) # convert to float (in case it's not)
    y1 = y[:traning_nb] # training output
    y2 = y[traning_nb:] # validation output
    return X1,y1, X2,y2

def sigmoid(x, theta):
    z = np.dot(x, theta)
    return 1/(1+np.exp(-z))


def hypothesis(theta, x):
    return sigmoid(x, theta)


def cost_function(theta, x, y, n):
    h = hypothesis(theta, x)
    return -(1/n)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))


def gradient(theta, X, y, n):
    h = hypothesis(theta, X)
    return (1/n) * np.dot(X.T, (h-y))

def predict(h):
    h1 = []
    for i in h:
        if i>=0.5:
            h1.append(1)
        else:
            h1.append(0)
    return h1




def train(Xt, yt, Xv, yv, nt, nv, epoch, alpha, graph=True, disp=True):
    """
    Function that train the logistic regression model

    PARAMETERS : 

        - Xt : Training set            ¤     - Xv : Validation set
        - yt : Training set targets    ¤     - yv : Validation set targets
        - nt : number of training set  ¤     - nv : number of validation set
        - epoch                        ¤     - alpha : param of gradient descent
        - graph : display the graphs
    
    OUTPUT : 

        theta : vector of the optimal weights

    PROCESS : 

        for each epoch :
            we compute the new weigths (theta) from the training set 
            we compute the loss for the training set
            we compute the accuracy for the training set 
            we use the weights from training set to compute loss and accuracy of the validation set
        
        we return the final weights for the prediction
    """

    ###########################################################
    #####                SETTING VARIABLES                #####
    ###########################################################
    training_err = []
    validation_err = []
    training_acc_1 = []
    training_acc_0 = []
    validation_acc_1 = []
    validation_acc_0 = []
    last_t_loss = None
    theta = np.zeros((Xt.shape[1], 1))
    nbt_1 = sum(yt)[0]
    nbt_0 = len(yt) - nbt_1
    nbv_1 = sum(yv)[0]
    nbv_0 = len(yv) - nbv_1   

    ###########################################################
    #####                 BEGIN TRAINING                  #####
    ###########################################################
    for e in tqdm(range(1, epoch + 1)):

        grad = gradient(theta,Xt,yt, nt)   # gradient computation
        theta = theta - alpha * grad * cost_function(theta, Xt, yt, nt)   # updating weights
        
        loss_t = cost_function(theta, Xt, yt, nt)  # compute loss with the current weight (training set)
        training_err.append(loss_t)

        out_t = hypothesis(theta, Xt)    # sigmoid application with the current weight (training set)
        yt_pred = predict(out_t)          # prediction h > 0.5 or h < 0.5
        accuracy_1 = 0
        accuracy_0 = 0                  # compute accuracy with the current weight (training set)
        for i in range(nt): 
            if yt_pred[i] == yt[i] and yt[i] == 1:
                accuracy_1 += 1
            rd_acc_1 = round(accuracy_1/nbt_1*100,4)

            if yt_pred[i] == yt[i] and yt[i] == 0:
                accuracy_0 += 1
            
            rd_acc_0 = round(accuracy_0/nbt_0*100,4)

        training_acc_1.append(rd_acc_1)
        training_acc_0.append(rd_acc_0)

        loss_v = cost_function(theta, Xv, yv, nv)  # compute loss with the current weight (validation set)
        validation_err.append(loss_v)
        
        
        out_v = hypothesis(theta,Xv)  # sigmoid application with the current weight (validation set)
        yv_pred = predict(out_v)     # prediction h > 0.5 or h < 0.5
        accuracy_0 = 0
        accuracy_1 = 0                 # compute accuracy with the current weight (validation set)
        for i in range(nv):
            if yv_pred[i] == yv[i] and yv[i] == 1:
                accuracy_1 += 1
            rd_acc_11 = round(accuracy_1/nbv_1*100,4)
            
            if yv_pred[i] == yv[i] and yv[i] == 0:
                accuracy_0 += 1
            rd_acc_00 = round(accuracy_0/nbv_0*100,4)
        
        validation_acc_1.append(rd_acc_11)
        validation_acc_0.append(rd_acc_00)

        if (e % (epoch / 10) == 0 or e == epoch) and disp == True:
            print(f'\n¤¤¤¤¤¤¤¤¤¤¤¤ EPOCH {e} ¤¤¤¤¤¤¤¤¤¤¤¤')
            if last_t_loss and last_t_loss < loss_t:
                print('   >>>>> LOSS INCREASING <<<<<   ')
            else:
                print(f' - Training loss : {round(loss_t,4)}')
                print(f' - Validation loss : {round(loss_v,4)}')
                print(f' - Loss difference : {round(((loss_t - loss_v)/loss_t),4)} % ')
                print(f' - Training accuracy : {training_acc[-1]} %')
                print(f' - Validation accuracy : {validation_acc[-1]} %')
                print(f'¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')
            
            last_t_loss = loss_t
    
    print(f'\n accuracy of 0 (training) {rd_acc_0}')
    print(f'\n accuracy of 1 (training) {rd_acc_1}')
    print(f'\n accuracy of 0 (validation) {rd_acc_00}')
    print(f'\n accuracy of 0 (validation) {rd_acc_11}')

    if disp == True:
        print(f'\n¤¤¤¤¤¤¤¤¤¤¤¤  OPTIMAL THETA ¤¤¤¤¤¤¤¤¤¤¤¤\n')
        for i in range(theta.shape[0]):
            print(f'w{i} = {theta[i][0]}')




    if graph == True:
        ###########################################################
        #####                    PLOT LOSS                    #####
        ###########################################################
        plt.figure()
        plt.title('Loss')
        plt.plot(training_err, 'dodgerblue', label='Training loss')
        plt.plot(validation_err, 'darkorange', label='Validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        ###########################################################
        #####                  PLOT ACCURACY                  #####
        ###########################################################
        plt.figure()
        plt.title('Accuracy')
        plt.plot(training_acc_0, 'dodgerblue', label='Training specificity')
        plt.plot(validation_acc_0, 'darkorange', label='Validation specificity')
        plt.plot(training_acc_1, 'red', label='Training recall')
        plt.plot(validation_acc_1, 'green', label='Validation recall')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

        ###########################################################
        #####             PLOT SMOOTHED ACCURACY              #####
        ###########################################################
        acc_v_smooth = savgol_filter(validation_acc, 70, 4) 
        acc_smooth = savgol_filter(training_acc, 70, 4) 
        plt.figure()
        plt.title('Smoothed Accuracy')
        plt.plot(acc_smooth, 'dodgerblue', label='Training accuracy')
        plt.plot(acc_v_smooth, 'darkorange', label='Validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    return theta, training_err, validation_err


###################################

path = 'C:\\Users\\Public\\Documents\\IPSA\\SEOULTECH\\COURSES\\MACHINE LEARNING\\datasets\\pulsar_data_train.csv'





training_set, training_output, validation_set, validation_output = init_frame(path)


n_training = training_set.shape[0]
n_validation = validation_set.shape[0]

alpha = 0.001
epoch = 100
w = train(training_set,training_output,validation_set, validation_output, n_training, n_validation, epoch, alpha,True, False)

    

