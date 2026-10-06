import numpy as np


import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('ex1data1.txt', header = None, delimiter = ",") #read from dataset
X = data.iloc[:,0] # read first column, will be put in a 'series' variable
print('X.shape: ', X.shape)
y = data.iloc[:,1] # read second column, will be put in a 'series' variable
print('y.shape: ', y.shape)
m = len(y) # number of training example (97)
print('Number of samples:', m)
print(data.head()) # view first few rows of the data



plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


print(type(X)) # should be a pd.Series()
X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
y = y.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray

# Will generate a warning:
# Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.

# # Better w/o warning
# X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
# y = y.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray

theta = np.zeros([2,1]) # start off with a (0, 0) array
iterations = 100000 # 
learning_rate = 0.01 # magic number, defines the increase in change in Theta in every iteration we take



# CALCULATE COST

X_ones = np.hstack([np.ones((m, 1)), X]) # stack colom of ones with column of the x values

def computeCost(X_ones, y, theta):
    m = len(y) 

    predictions =  np.dot(X_ones, theta) # multiplu x matrix with theta
    errors = predictions - y

    cost = (1/(2*m)) * np.sum(errors**2)        

    return cost


print( computeCost(X_ones, y, theta))




# GRADIENT DESCENT 

iterations = 100000


def gradientDescent(X_ones, y, theta, learning_rate, iterations):

    m = len(y)
    for i in range(iterations):
        predictions = np.dot(X_ones, theta) # make matrix off all predictions based on current theta
        errors = predictions - y # alc error 
        gradient = (1/m) * np.dot(X_ones.T, errors) #calc gradient of funct based on current theta 
        
        

        theta = theta - learning_rate * gradient # new theta takes step of gradient descent

    print(gradient)
    return theta







theta_refined = gradientDescent(X_ones, y, theta, learning_rate, iterations)
cost_refined =  computeCost(X_ones, y, theta_refined)
print(cost_refined)


iterations = 100000
learning_rate = 0.01

theta_refined = gradientDescent(X_ones, y, theta, learning_rate, iterations)
cost_refined =  computeCost(X_ones, y, theta_refined)
print(cost_refined)




