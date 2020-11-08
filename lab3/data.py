#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import keras

# Loading the data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
# Just seeing one image
# %%
plt.imshow(x_train[1])
plt.imshow(x_train[2])
plt.imshow(x_train[3])
# %%
# Data normalization 
# One  normalizes the data so they are approximetely the same scale.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# %%
# Converting the train labels to one-hot encoding.
y_train_hot_key = keras.utils.to_categorical(y_train, num_classes = 10)
y_train = y_train_hot_key
# %% 
# Splitting the training data into training and validating.
# - Training data — used for training the model
# - Validation data — used for tuning the hyperparameters and evaluate the models
x_train, x_validate, y_validate, y_validate = train_test_split(x_train, y_train, test_size=0.2)
# %%
# Reshaapping the datasets
x_train = np.expand_dims(x_train, axis=3)
x_validate = np.expand_dims(x_validate, axis=3)
# %%
