from os import waitstatus_to_exitcode
from turtle import TurtleScreenBase

import numpy as np

#This function (sigmoid) will convert our number x to a number from the interval (0,1)
def sigmoid(x):
    return 1/(1+np.exp(-x)) #np.exp(-x) - exponent, the number e raised to the power -x

#Inputs for which results are known
inputs_for_training = np.array([[0,0,1],
                            [1,1,0],
                            [1,0,1],
                            [1,1,1]])
#Outputs for our data. 
outputs_of_training = np.array([[0,1,1,1]]).T # .T - transposes our matrix from 1x4 to a 4x1 matrix

np.random.seed(1) #Initiating a random seed

ai_weights = np.random.random((3,1)) * 2 - 1

turns_of_training = 1000000 #How many time our AI will adjust the weights. More turns = better accuracy 

print("Our start weights: ")
print(ai_weights)

print("-------------------------------------------")

for i in range (turns_of_training):
    layer_of_inputs = inputs_for_training
    layer_of_outputs = sigmoid(np.dot(layer_of_inputs,ai_weights)) #np.dot(A,B) - Multiplies matrix A by B
    
    error = outputs_of_training-layer_of_outputs #Error, how much our result differs from the expected one
    
    #We are looking for the adjustment that is needed to correct the weights
    adjustments = np.dot(layer_of_inputs.T, error * (layer_of_outputs * (1 - layer_of_outputs))) 
    
    ai_weights+=adjustments #Adjusting our weights

print("Our end weights: ")
print(ai_weights)

print("-------------------------------------------")

print("Our results after training: ")
print(layer_of_outputs)

print("-------------------------------------------")

new_input=[1,0,0]
new_output=sigmoid(np.dot(new_input,ai_weights))

print("Results for the new input: ")
print(new_output)