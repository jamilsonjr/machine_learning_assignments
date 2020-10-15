import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from functions import *

## Part 1 Linear Regression
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
plt.legend()
plt.grid(True)
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
# Training
beta_hat = ls_fit_1d(data2_x, data2_y, 2)
x = np.linspace(-1,1,100)
y =  beta_hat[0][2]*x ** 2 + beta_hat[0][1]*x + beta_hat[0][0]
plt.plot(x, y, '-r', label='LS fit')
plt.legend()
plt.grid(True)
plt.show()

print('\n The coeficients are:\n beta_0 =',beta_hat[0][0],'\n beta_1=', beta_hat[0][1],'\n beta_2=', beta_hat[0][2])

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
print('\nSHAPE DATA:\n data_x:', np.shape(data2a_x), 'data_y',np.shape(data2a_y))

# Reshaping data
data2a_y = np.reshape(data2a_y, (50, 1))
print('\n Shape of data after reshape:', np.shape(data2a_y))

# Plot the data 
plt.scatter(data2a_x, data2a_y, label='data')
beta_hat = ls_fit_1d(data2a_x, data2a_y, 2)
x = np.linspace(-1,1,100)
y =  beta_hat[0][2]*x ** 2 + beta_hat[0][1]*x + beta_hat[0][0]
plt.plot(x, y, '-r', label='LS fit')
print('\n The coeficients are:\n beta_0 =', beta_hat[0][0],'\n beta_1=', beta_hat[0][1],'\n beta_2=', beta_hat[0][2])
plt.legend()
# Show the plot
plt.grid(True)
plt.show()

# Calculate the SSE
X = beta_hat[1]
# Prediction
y_hat = np.matmul(X, beta_hat[0])
# Sum of Squared Errors
sse = np.linalg.norm(data2a_y-y_hat)
print('\n The SSE is:', sse)
# Sum of Squared Erros without the outliers
for i in range(len(data2a_y)-2):
    if(np.absolute(y[i]-data2a_y[i]) > 1.5):
        data2a_x = np.delete(data2a_x, i)
        data2a_y = np.delete(data2a_y, i)
        y = np.delete(y, i)
# Reshaping data 
data2a_y = np.reshape(data2a_y, (48, 1))
data2a_x = np.reshape(data2a_x, (48, 1))
# SSE without outliers
beta_hat = ls_fit_1d(data2a_x, data2a_y, 2)
X = beta_hat[1]
# Prediction
y_hat = np.matmul(X, beta_hat[0])
# Sum of Squared Errors
sse = np.linalg.norm(data2a_y-y_hat)
print('\n The SSE without outliers is:', sse)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
## Part 2 Lasso and Ridge
data3_x = np.load('data3_x.npy')
data3_y = np.load('data3_y.npy')

# Instantiation Ridge and Lasso classes 
rr_control = Ridge(alpha=0.01)
lr_control = Lasso(alpha=0.01)
# Fitting the data 
rr_control.fit(data3_x, data3_y)
lr_control.fit(data3_x, data3_y)
# For all values in the range
n_alphas = 1000
alphas = np.linspace(0.001, 10, n_alphas)
rr_coefs = []
best_alpha = []
for alpha in alphas:
    rr = Ridge(alpha=alpha, fit_intercept=False)
    lr = Lasso(alpha=alpha, fit_intercept=False)
    rr.fit(data3_x, data3_y)
    lr.fit(data3_x, data3_y)
    try:
        rr_coefs = np.vstack((rr_coefs, rr.coef_))
        lr_coefs = np.vstack((lr_coefs, lr.coef_))
        if(0 in lr.coef_):
            best_alpha.append(alpha) # Sim eu sei que é abuso de memoria, TODO: melhorar... sorry
    except:
        rr_coefs = rr.coef_
        lr_coefs = lr.coef_
# Plotting figures
for i in range(1000):
    rr_control.coef_ = np.vstack((rr_control.coef_, rr_control.coef_))
    lr_control.coef_ = np.vstack((lr_control.coef_, lr_control.coef_))

print('\n Seeing shit!\n', np.shape(rr_control.coef_))
plt.figure()
## plt.plot(alphas, np.transpose(rr_control.coef_), label='[LS] feature 0')
plt.plot(alphas, np.transpose(rr_coefs)[0], label='feature 0')
plt.plot(alphas, np.transpose(rr_coefs)[1],  label='feature 1')
plt.plot(alphas, np.transpose(rr_coefs)[2],  label='feature 2')
plt.title('Ridge')
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(alphas, np.transpose(lr_coefs)[0], label='feature 0')
plt.plot(alphas, np.transpose(lr_coefs)[1],  label='feature 1')
plt.plot(alphas, np.transpose(lr_coefs)[2],  label='feature 2')
plt.title('Lasso')
plt.xscale('log')
plt.grid(True)
plt.xlabel(r'$\alpha$')
plt.legend()
plt.show()
# PROF: Porque no lasso os 3 vão para zero?
# Choose the adequate value for alpha
# We want an alpha that eliminates one featuew but does not eliminate the other features
print('The best alpha is', best_alpha[0])
# Doing the y_hat with respect to alpha
lr = Lasso(alpha=0.071, fit_intercept=False)
lr.fit(data3_x, data3_y)
ls = Lasso(alpha=0, fit_intercept=False)
ls.fit(data3_x, data3_y)
y_hat_lr = lr.predict(data3_x)
y_hat_ls = lr.predict(data3_x)
plt.figure()
plt.plot(np.linspace(0, 50, 50), y_hat_lr, label='Lasso Prediction')
plt.plot(np.linspace(0, 50, 50), y_hat_ls, label='Least Squares Prediction')
plt.grid(True)
plt.legend()
plt.show()
plt.show()
y_hat_lr = np.reshape(y_hat_lr, (50,1))
y_hat_ls = np.reshape(y_hat_ls, (50,1))
# Calculating the SSE
sse_lr = np.linalg.norm(data3_y-y_hat_lr)
sse_ls = np.linalg.norm(data3_y-y_hat_ls)
print('\n The SSE for the Least Squares Prediction is:', sse_ls, '\n The SSE of the Lasso Prediction is:', sse_lr )
print(y_hat_lr.shape, data3_y.shape)


# TODO Ver a cena do ruido. Mandei mail à stora

