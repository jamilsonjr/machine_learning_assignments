import numpy as np
import matplotlib.pyplot as plt
from functions import *


## About data 1
print('\nAbout data1:\n')
# Import the data
data1_x = np.load('data1_x.npy')
data1_y = np.load('data1_y.npy') 
# Plot the data
plt.scatter(data1_x, data1_y, label='data')
# LS_fit of data1
beta_hat = ls_fit_1d(data1_x, data1_y, 1)
x = np.linspace(-1,1,100)
y = beta_hat[0][1][0]*x + beta_hat[0][0][0]
plt.plot(x, y, '-r', label='LS fit')
# Add a legend
plt.legend()
# Show the plot
plt.show()
# Printing the coeficients
print('\n The coeficients are:\n beta_0 =',beta_hat[0][0],'\n beta_1=', beta_hat[0][1])
# Calculate the SSE 
X = beta_hat[1]
# Prediction
y_hat = np.matmul(X, beta_hat[0])
# Sum of Squared Errors
sse = np.linalg.norm(data1_y-y_hat)
print('\n The SSE is:', sse)

# ---------------------------------------------------------------------------- #
## About data2
print('\nAbout data2:\n')
# Import the data
data2_x = np.load('data2_x.npy')
data2_y = np.load('data2_y.npy') 
# Plot the data 
plt.scatter(data2_x, data2_y, label='data')

# beta_hat = ls_fit_1d(data2_x, data2_y, 2)
beta_hat = ls_fit_1d(data2_x, data2_y, 2)
x = np.linspace(-1,1,100)
y =  beta_hat[0][2]*x ** 2 + beta_hat[0][1]*x + beta_hat[0][0]
plt.plot(x, y, '-r', label='LS fit')
print('\n The coeficients are:\n beta_0 =',beta_hat[0][0],'\n beta_1=', beta_hat[0][1],'\n beta_2=', beta_hat[0][2])

# Add a legend
plt.legend()
# Show the plot
plt.show()

# Calculate the SSE
X = beta_hat[1]
# Prediction
y_hat = np.matmul(X, beta_hat[0])
# Sum of Squared Errors
sse = np.linalg.norm(data2_y-y_hat)
print('\n The SSE is:', sse)

# ---------------------------------------------------------------------------- #
## About data2a
print('\nAbout data2a:\n')
# Import the data
data2a_x = np.load('data2a_x.npy')
data2a_y = np.load('data2a_y.npy') 
# Plot the data 
plt.scatter(data2a_x, data2a_y, label='data')

# beta_hat = ls_fit_1d(data2_x, data2_y, 2)
beta_hat = ls_fit_1d(data2a_x, data2a_y, 2)
x = np.linspace(-1,1,100)
y =  beta_hat[0][2]*x ** 2 + beta_hat[0][1]*x + beta_hat[0][0]
plt.plot(x, y, '-r', label='LS fit')
print('\n The coeficients are:\n beta_0 =',beta_hat[0][0],'\n beta_1=', beta_hat[0][1],'\n beta_2=', beta_hat[0][2])
# Add a legend
plt.legend()
# Show the plot
plt.show()

# Calculate the SSE
X = beta_hat[1]
# Prediction
y_hat = np.matmul(X, beta_hat[0])
# Sum of Squared Errors
sse = np.linalg.norm(data2_y-y_hat)
print('\n The SSE is:', sse)