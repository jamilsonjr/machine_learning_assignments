# %% Importing and analysing the data
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


X_train = pd.DataFrame(np.load('Real_Estate_Xtrain.npy'))
y_train = pd.DataFrame(np.load('Real_Estate_ytrain.npy'))
X_test = pd.DataFrame(np.load('Real_Estate_Xtest.npy'))
y_test = pd.DataFrame(np.load('Real_Estate_ytest.npy'))
print(X_train.shape, y_train.shape)
# Scaling features
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
#%% define base model
def create_model(neurons):
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#%% Create the model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# %% Grid Search
# define the grid search parameters
neurons = [2, 4, 6, 8, 16, 24, 26]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# %% Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# %% Testing
# Create the best model for testing
def create_final_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model
regressor = KerasRegressor(build_fn=create_final_model,epochs=100,batch_size=10)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
prediction = np.reshape(prediction, (102,1))
mse = ((y_test - prediction)**2).mean()
print(mse)
max_squared_error = (np.amax(y_test - prediction)) ** 2
print(max_squared_error) 


# %% Another learned Regression Method: LAsso Regression
# Discover the best value for alpha
n_alphas = 1000
alphas = np.linspace(0.001, 10, n_alphas)
best_alpha = []
best_alpha = []
for alpha in alphas:
    lr = Lasso(alpha=alpha, fit_intercept=True)
    lr.fit(X_train, y_train)
    try:
        lr_coefs = np.vstack((lr_coefs, lr.coef_))
        if(0 in lr.coef_):
            best_alpha.append(alpha) 
    except:
        lr_coefs = lr.coef_
best_alpha = best_alpha[0]
print('The best alpha is', best_alpha)
#%% The the model for the best model for alpha 
lr = Lasso(alpha=best_alpha, fit_intercept=True)
lr.fit(X_train, y_train)
#%% Testing and evaluating the model
prediction = lr.predict(X_test)
prediction = np.reshape(prediction, (102,1))
mse = ((y_test - prediction)**2).mean()
print(mse)
max_squared_error = (np.amax(y_test - prediction)) ** 2
print(max_squared_error) 

# %%
