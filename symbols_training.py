import numpy as np
import pandas as pd
import cv2
import pickle 
from datetime import datetime

# get starting time
start_time = datetime.now()

layersize = 25
alpha = 0.08
iterations = 2500
n_hidden = 3


symbols = ['plus cleaned', 'minus', 'times', 'div']
    
with open('plus cleaned_images.txt', 'rb') as a:
    add = pickle.load(a)
    
with open('minus_images.txt', 'rb') as m:
    minus = pickle.load(m)

with open('times_images.txt', 'rb') as t:
    times = pickle.load(t)

with open('div_images.txt', 'rb') as d:
    div = pickle.load(d)

add = add.T
minus = minus.T
times = times.T
div = div.T

data = np.column_stack([add, minus, times, div])
data = data.T
np.random.shuffle(data)
data = data.T    
labels = data[-1]
labels = labels.astype(int)
data = data[:-1]
data = data.T
(samplesize, picturesize) = data.shape


def init_params(n_hidden):
    #W for weights, b for bias
    weights = []
    bias = []
    
    # make an array with the correct number of weights and biases depending on n_hidden
    for i in range(n_hidden + 1):
        weights.append('')
        bias.append('')
    
    # randomizing weights and biases for the first layer with the correct size
    weights[0] = np.random.rand(layersize, picturesize) - 0.5
    bias[0] = np.random.rand(layersize,1) - 0.5
    
    # ranomizing weights and biases for hidden layers
    for j in range(n_hidden - 1):
        weights[1 + j] = np.random.rand(layersize,layersize) - 0.5 
        bias[1 + j] = np.random.rand(layersize,1) - 0.5
    
    # randomizing weights and biases for last layer of weights (need to have a differnt size for output layer)
    weights[-1] = np.random.rand(4,layersize) - 0.5 
    bias[-1] = np.random.rand(4,1) - 0.5
    return weights, bias 

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def dsigmoid(Z):
    return (np.exp(-Z)) / (1 + np.exp(-Z))**2

def forward_prop(n_hidden, weights, bias, X):
    
    z_layer = []
    activations = []
    
    for i in range(n_hidden+1):
        z_layer.append('')
        activations.append('')
    
    # calculate fist layer of forward prop given the image it is reading
    z_layer[0] = weights[0].dot(X.T) + bias[0]
    activations[0] = sigmoid(z_layer[0])
    
    # calculate forward prop for all layers (excluding first)
    for j in range(n_hidden):
        z_layer[j + 1]  = weights[j + 1].dot(activations[j]) + bias[j + 1]
        activations[j + 1] = sigmoid(z_layer[j + 1])
    
    return z_layer, activations

def one_hot(Y):
    # add one at the index of the max to an array of xeros
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(n_hidden, z_layer, activations, weights, X, Y):
    
    z_layer = z_layer[::-1]
    activations = activations[::-1]
    weights = weights[::-1]
    
    # d_z = [dZ3, dZ2, dZ1]
    d_z = []
    d_w = []
    d_b = []
    
    for i in range(n_hidden + 1): 
        d_z.append('')
        d_w.append('')
        d_b.append('')
    #m is the number of pictures being trained on
    #cost(Y) will find who got it right
    
    cost_Y = one_hot(Y)
    #dZ2 will subtract the ones that got it right from 
    #the final layer
    d_z[0] = activations[0] - cost_Y
    d_w[0] = 1/samplesize * d_z[0].dot(activations[1].T)
    d_b[0] = 1/samplesize * np.sum(d_z[0])
    
    for j in range(n_hidden-1): 
        d_z[j + 1] = weights[j].T.dot(d_z[j]) * dsigmoid(z_layer[j + 1])
        d_w[j + 1] = 1 / samplesize * d_z[j + 1].dot(activations[j + 2].T)
        d_b[j + 1] = 1 / samplesize * np.sum(d_z[j + 1])
    
    d_z[-1] = weights[-2].T.dot(d_z[-2]) * dsigmoid(z_layer[-1])
    #dW1 is the dot product of dZ1 and the current image
    #divided by total number of images
    d_w[-1] = 1 / samplesize * d_z[-1].dot(X)
    #db1 finds the average of dZ1 and uses it as a bias
    d_b[-1] = 1 / samplesize * np.sum(d_z[-1])
    
    d_w = d_w[::-1]
    d_b = d_b[::-1]
    return d_w, d_b, d_z

def update_params(weights, bias, d_w, d_b, alpha):
    for i in range(len(weights)):
        weights[i] = weights[i] - alpha * d_w[i]
        bias[i] = bias[i] - alpha * d_b[i]
        
    return weights, bias

def get_predictions(last_a):
    return np.argmax(last_a,0)

def get_accuracy(predictions,Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(n_hidden, X, Y, alpha, iterations, layersize):
    weights, bias = init_params(n_hidden)
    for i in range(iterations):
        z_layer, activations = forward_prop(n_hidden, weights, bias, X)
        d_w, d_b, d_z  = backward_prop(n_hidden, z_layer, activations, weights, X, Y)
        weights, bias = update_params(weights, bias, d_w, d_b, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(activations[-1])
            print(get_accuracy(predictions,Y))
            
        if i % 1000000 == 0 and i != 0: 
            for j in range(len(activations[-1])): 
                print("Chance of " + str(j) + ':  ' + str(round(activations[-1][j][i]*100,2)) + '%')
            csv_to_image(data[i])    
            
    return weights, bias, activations[-1], Y, activations, d_z
  
            
def csv_to_image(csv_row):
    img = csv_row.reshape(135,155)
    image = np.zeros((135,155,1))
    image[:,:,0] = img
    # image = cv2.resize(image, (310,270),10,10)   
    cv2.imshow("image",image)
    # wait untill window is closed
    cv2.waitKey(0)

def display_time(run_time):
    run_time = str(run_time)
    print()
    print('Run time:')
    print(run_time[0] + 'h' + ' ' + run_time[2:4] + 'm' + ' ' + run_time[5:] + 's')


weights, bias, last_a, Y, activations, d_z = gradient_descent(n_hidden, data, labels, alpha, iterations, layersize)




trained_model = [weights, bias, n_hidden]     
    
with open('symbols_training_3hidden_i2500_L25_a0.08.txt', 'wb') as f:
    pickle.dump(trained_model,f)
    
end_time = datetime.now()

run_time = end_time - start_time

display_time(run_time)        
    
        

    


