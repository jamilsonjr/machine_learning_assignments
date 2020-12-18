#%% Importing Libraries and Data

# importing libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# importing data
x = np.load('Cancer_Xtrain.npy')
y = np.load('Cancer_ytrain.npy')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
# Scaling features
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# %% Training and validation the data with several kernels

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                    {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.1, 1, 10, 100, 100]}
                    ]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#%% Retrain the data for the best params:
final_model = SVC(C = 1, kernel='poly', degree=4)
final_model.fit(X_train, y_train)


#%% Testing with unseen data
x = np.load('Cancer_Xtest.npy')
y_test = np.load('Cancer_ytest.npy')
# Scaling data
X_test = min_max_scaler.fit_transform(x)
y_true, y_pred = y_test, final_model.predict(X_test)
print(classification_report(y_true, y_pred))
# -----------------------------------------------------------------------------------------------------------------
# %% Another classification method learned: K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
# %% Training and testing the k-nn classifier with several values for k, in order to find the best one

# importing data
x = np.load('Cancer_Xtrain.npy')
y = np.load('Cancer_ytrain.npy')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Training and testing with several values for k
tuned_parameters = [{'n_neighbors': [3, 5, 7, 11, 13, 17, 19, 23]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(
        KNeighborsClassifier(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Training model for the best parameters
final_model = KNeighborsClassifier(n_neighbors=3)
final_model.fit(X_train, y_train)



# %%
#%% Testing with unseen data
x = np.load('Cancer_Xtest.npy')
y_test = np.load('Cancer_ytest.npy')
# Scaling data
X_test = min_max_scaler.fit_transform(x)
y_true, y_pred = y_test, final_model.predict(X_test)
print(classification_report(y_true, y_pred))
# %%
