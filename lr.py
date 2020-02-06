import numpy as np
import math
import sys

#unpacking the arguments 
train_input,validation_input,test_input,dict_input,train_out,test_out,metrics_out,num_epoch = sys.argv[1:]

# dot product 
def dot(W,X):
    product = 0.0
    for i,v in X.items():
        product += v*W[i]
    return(product)

# sigmoid function
def sigmoid(x):
    z = math.exp(x)
    return(z/(1+z))

# one step SGD
def one_step_SGD(params,rate,example):
    y,x = example[0],example[1]
    constant = sigmoid(dot(params,x)) - y
    for j,v in x.items():
        gradient = constant*v
        params[j] -= rate*gradient
    return(params)

def binary_logistic_train(data,num_epoch,rate):
    params = {}
    params[-1] = 0 #bias term 
    for i in vocab.values():
        params[i] = 0
    for n in range(num_epoch):
        for example in data:
            params = one_step_SGD(params,rate,example)
    return(params)

def binary_logistic_predict(data,params):
    y_predicted = []
    for example in data:
        prob_y1 = sigmoid(dot(params,example[1]))
        prob_y0 = 1 - prob_y1
        if prob_y1 >= prob_y0:
            y_predicted.append(1)
        else:
            y_predicted.append(0)
    return(y_predicted)

def errorRate(labels,predicted_labels):
    error = 0
    for i in range(len(labels)):
        if (labels[i] != predicted_labels[i]):
            error += 1
    return(error/len(labels))


# getting the data
# dictionary
f = open(dict_input,'r')
vocab = [(x,int(y)) for x,y in (line.strip().split(" ") for line in f)]
vocab = dict(vocab)
f.close()

def read_file(filename):
    f = open(filename,'r')
    data = [[y,x] for y,*x in (line.strip().split("\t") for line in f)]
    f.close()
    return(data)

def create_feature_matrix(data):
    feature_matrix = []
    for i in range(len(data)):
        feature_vector = dict([(int(x),int(y)) for x,y in (z.split(':') for z in data[i][1])])
        feature_vector[-1] = 1 #bias term 
        feature_matrix.append(feature_vector)
    return(feature_matrix)
    
# training data
formatted_train = read_file(train_input)
labels_train = [int(y) for y,x in formatted_train]
features_train = create_feature_matrix(formatted_train)
data_train = list(zip(labels_train,features_train))
                  
# validation data
# formatted_validation =read_file(valid_input)
# labels_validation =[int(y) for y,x in formatted_validation]
# features_valid = create_feature_matrix(formatted_valid)
# data_valid = list(zip(labels_valid,features_valid)

# test data
formatted_test = read_file(test_input)
labels_test  = [int(y) for y,x in formatted_test]
features_test = create_feature_matrix(formatted_test)
data_test = list(zip(labels_test,features_test))


# training and predicting
params_mle = binary_logistic_train(data_train,int(num_epoch),0.1)
                 
labels_predicted_train = binary_logistic_predict(data_train,params_mle)
labels_predicted_test = binary_logistic_predict(data_test,params_mle)                 
                 
#Calculating training and testing error
error_train = errorRate(labels_train,labels_predicted_train)
error_test = errorRate(labels_test,labels_predicted_test)

#Output
prediction_train = [str(x)+'\n' for x in labels_predicted_train]
prediction_test = [str(x)+'\n' for x in labels_predicted_test]
open(train_out,'w').writelines(prediction_train)
open(test_out,'w').writelines(prediction_test)
open(metrics_out,'w').write('error(train):'+str(error_train)+'\n'+'error(test):'+str(error_test))
    
    