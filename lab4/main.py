# TODO: Perguntar a cena da função de custo.
# A simple example
#%% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from math import pi
from math import exp
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#%% Load
xtrain = np.load('data1_xtrain.npy')
ytrain = np.load('data1_ytrain.npy')
xtest = np.load('data1_xtest.npy')
ytest = np.load('data1_ytest.npy')
plt.figure()
for i in range(0,ytrain.size):
    if(ytrain[i]==1):plt.scatter(xtrain[i][0],xtrain[i][1],color='red')
    if(ytrain[i]==2):plt.scatter(xtrain[i][0],xtrain[i][1],color='green')
    if(ytrain[i]==3):plt.scatter(xtrain[i][0],xtrain[i][1],color='blue')
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.title('Training Data')
plt.grid()
plt.figure()
for i in range(0,ytrain.size):
    if(ytest[i]==1):plt.scatter(xtest[i][0],xtest[i][1],color='red')
    if(ytest[i]==2):plt.scatter(xtrain[i][0],xtest[i][1],color='green')
    if(ytest[i]==3):plt.scatter(xtest[i][0],xtest[i][1],color='blue')
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.title('Testing Data')
plt.grid()
# %% Naives Bayes Classifier
#%%
# Separate the features 
def separate_classes(data):
    separated = dict()
    for i in range(len(data)):
        row  = data[i]
        class_of_data = row[-1]
        if (class_of_data not in separated):
            separated[class_of_data] = list()
        separated[class_of_data].append(row[:-1])
    return separated
dataset=np.concatenate((xtrain, ytrain), axis=1)
separated = separate_classes(dataset)
# xtrain_x0_1 = []
# xtrain_x0_2 = []    
# xtrain_x0_3 = []
# xtrain_x1_1 = []
# xtrain_x1_2 = []
# xtrain_x1_3 = []

# for i in range(0, xtrain[:,0].size):
#     if(ytest[i]==1):xtrain_x0_1.append(xtrain[i,0])
#     if(ytest[i]==2):xtrain_x0_2.append(xtrain[i,0])
#     if(ytest[i]==3):xtrain_x0_3.append(xtrain[i,0])
#     if(ytest[i]==1):xtrain_x1_1.append(xtrain[i,1])
#     if(ytest[i]==2):xtrain_x1_2.append(xtrain[i,1])
#     if(ytest[i]==3):xtrain_x1_3.append(xtrain[i,1])

# xtrain_x0_1_mean = np.mean(xtrain_x0_1)
# xtrain_x0_2_mean = np.mean(xtrain_x0_2)
# xtrain_x0_3_mean = np.mean(xtrain_x0_3)
# xtrain_x1_1_mean = np.mean(xtrain_x1_1)
# xtrain_x1_2_mean = np.mean(xtrain_x1_2)
# xtrain_x1_3_mean = np.mean(xtrain_x1_3)

# xtrain_x0_1_var = np.var(xtrain_x0_1)
# xtrain_x0_2_var = np.var(xtrain_x0_2)
# xtrain_x0_3_var = np.var(xtrain_x0_3)
# xtrain_x1_1_var = np.var(xtrain_x1_1)
# xtrain_x1_2_var = np.var(xtrain_x1_2)
# xtrain_x1_3_var = np.var(xtrain_x1_3)

#%% Calculate mean and variance
def calculate_mean_and_variance_by_class(data):
    var_and_mean = dict() 
    for key, value in data.items():
        var_and_mean[key] = [(np.mean(col), np.var(col)) for col in zip(*value)]
    return var_and_mean

var_and_mean = calculate_mean_and_variance_by_class(separated)

def my_norm(data_point, mean, var):
    exponent = exp(-((data_point-mean)**2 / (2 * var**2 )))
    prob = (1 / (sqrt(2 * pi) * var)) * exponent
    return prob
#%% Calculate probabilities
def calculate_probabilities_by_class(data, var_and_mean):
    output = list()
    for i in range(len(data)):
        label = None
        max_probability = -1
        for key, value in var_and_mean.items():
            mean_x0 = value[0][0]
            var_x0 = value[0][1]
            mean_x1 = value[1][0]
            var_x1 = value[1][1]
            prob_x0 = my_norm(data[i][0], mean_x0, var_x0)
            prob_x1 = my_norm(data[i][1], mean_x1, var_x1)
            probability = prob_x0 * prob_x1 * 0.3 # Class prior probability
            if probability > max_probability:
                max_probability = probability
                label = key
        output.append(label)
    return output


        

    # probabilities = dict()
    # x_0 = True
    # for key, values in data.items():
    #     x_0 = True
    #     mean_x0 = var_and_mean[key][0][0]
    #     var_x0 = var_and_mean[key][0][1]
    #     mean_x1 = var_and_mean[key][1][0]
    #     var_x1 = var_and_mean[key][1][1]
    #     probabilities[key] = list()
    #     for col in zip(*values):
    #         if(x_0):
    #             probabilities[key].append(norm.pdf(col,mean_x0,var_x0))
    #         else: 
    #             probabilities[key].append(norm.pdf(col,mean_x1, var_x1))
    #         x_0 = False
    return 

output = calculate_probabilities_by_class(xtest, var_and_mean)

# print(np.shape(norm.pdf(xtrain_x0_1, xtrain_x0_1_mean, xtrain_x0_1_var)))
# print(np.shape(norm.pdf(xtrain_x1_1, xtrain_x1_1_mean, xtrain_x1_1_var)))

# print(np.shape(norm.pdf(xtrain_x0_2, xtrain_x0_2_mean, xtrain_x0_2_var)))
# print(np.shape(norm.pdf(xtrain_x1_2, xtrain_x1_2_mean, xtrain_x1_2_var)))

# print(np.shape(norm.pdf(xtrain_x0_3, xtrain_x0_3_mean, xtrain_x0_3_var)))
# print(np.shape(norm.pdf(xtrain_x1_3, xtrain_x1_3_mean, xtrain_x1_3_var)))

# %% Plotting results
plt.figure()
for i in range(0,ytrain.size):
    if(output[i]==1):plt.scatter(xtest[i][0],xtest[i][1],color='red')
    if(output[i]==2):plt.scatter(xtrain[i][0],xtest[i][1],color='green')
    if(output[i]==3):plt.scatter(xtest[i][0],xtest[i][1],color='blue')
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.title('Testing Data Output')
plt.grid()
# %% Calculating Errors
naive_bayes_accuracy_score = accuracy_score(ytest, output)
print(naive_bayes_accuracy_score)
#  Practical Assignment
## Training
#%%  Loading data
# loading trigrams
pt = read_csv('pt_trigram_count.tsv', sep='\t', header=None)
fr = read_csv('fr_trigram_count.tsv', sep='\t', header=None)
es = read_csv('es_trigram_count.tsv', sep='\t', header=None)
en = read_csv('en_trigram_count.tsv', sep='\t', header=None)
print(fr)
# %% data format and content
pt_contents = pt.head()
pt_format = pt.shape

fr_contents = fr.head()
fr_format = fr.shape

es_contents = es.head()
es_format = es.shape

en_contents = en.head()
en_format = en.shape
print(fr_contents)

# %% Creating Xtrain and y_train
Xtrain = [pt.loc[:,2],fr.loc[:,2],es.loc[:,2],en.loc[:,2]]
print(Xtrain)
ytrain = ['pt', 'fr', 'es', 'en']
# %% Training the model
naive_bayes = MultinomialNB()
naive_bayes.fit(Xtrain, ytrain)

# %% Testing the model

predictions = naive_bayes.predict(Xtrain)
#evaluate
accuracy = accuracy_score(ytrain, predictions)
print(accuracy)

# %% From the table 
sentences = ['Que fácil es comer peras.','Que fácil é comer peras.','Today is a great day for sightseeing.','Je vais au cinéma demain soir.', 'Ana es inteligente y simpática.', 'Tu vais à escola hoje.']
# %% Create Vectorizer
vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='char', vocabulary=pt.loc[:,1])

# %% Learning data trigrams 
Xtest = vectorizer.fit_transform(sentences)
print(Xtest)


# %% Testing
y_test = naive_bayes.predict(Xtest)
print(y_test) 
# %% Predicting predicting classification margin 
prob = naive_bayes.predict_proba(Xtest)
print(prob)