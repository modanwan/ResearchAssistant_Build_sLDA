
# coding: utf-8

# In[ ]:


import os
import numpy as np
from slda.topic_models import SLDA
from scipy.sparse import csc_matrix

"""
    DIRECTORY: directory where you save your input and output files
    VOCAB: vocabulary data
    MULT_TRAIN:
    MULT_TEST:
    DEPVARS_TRAIN:
    DEPVARS_TEST:
"""
DIRECTORY = "/.../code/slda/temp/"
VOCAB = DIRECTORY + "vocab.dat"
MULT_TRAIN = DIRECTORY + "mult_train.dat"
MULT_TEST = DIRECTORY + "mult_test.dat"
DEPVARS_TRAIN = DIRECTORY + "depvars_train_adv.dat"
DEPVARS_TEST = DIRECTORY + "depvars_test_adv.dat"

ngrams1 = open(VOCAB).readlines()
ngrams = []
for ngram in ngrams1:
    ngram = ngram.strip()
    ngrams.append(ngram)

V = len(ngrams) # Vocabulary
K = 150 # number of topics
alpha = np.ones(K)
sigma2 = 0.25
nu2 = K

# Estimate parameters
_K = K
_alpha = alpha
_beta = np.repeat(0.01, V)
_mu = 0
_nu2 = nu2
_sigma2 = sigma2
n_iter = 200 # Number of iterations --> 200 iterations seemed to work fine for me

#### IMPORT TRAINING AND TEST DATA INTO MATRICES WITH DIMENSIONS: # DOCUMENTS x # NGRAMS

def import_data(filename):
    data1 = open(filename).readlines()
    D = len(data1)
    data = []
    rows = []
    cols = []
    for i,line in enumerate(data1):
        line = line.strip()
        pairs = line.split(' ')
        del pairs[0]
        for pair in pairs:
            col = int(pair.split(':')[0].strip())
            val = int(pair.split(':')[1].strip())
            data.append(val)
            cols.append(col)
            rows.append(i)
    outputmatrix = csc_matrix((data, (rows, cols)), shape=(D, V))
    return outputmatrix


X = import_data(MULT_TRAIN)
X_test = import_data(MULT_TEST)

filenames = open(DEPVARS_TRAIN).readlines()
y = []
for filename in filenames:
    depvar = float(filename.split('|')[1].strip())
    y.append(depvar)

filenames_test = open(DEPVARS_TEST).readlines()
y_test = []
for filename in filenames_test:
    depvar = float(filename.split('|')[1].strip())
    y_test.append(depvar)

slda = SLDA(_K, _alpha, _beta, _mu, _nu2, _sigma2, n_iter, seed=42)


slda.fit(X, y)

burn_in = max(n_iter - 100, int(n_iter / 2))
eta1 = slda.eta[burn_in:] # The burn_in just ignores the first several iterations as the program is still trying to figure things out
eta_pred = eta1.mean(axis=0) # These are the topic coefficients --> (mean(axis=0) just means to take the mean of each column in eta


#### WRITE TOPICS TO FILE ####

file_topics = open(DIRECTORY + 'pytopics_adv1.txt', 'w')

topic_distributions = slda.phi
n_top_words = 30 # This is how many words you want to show up in each topic in the output file
n_top_words = n_top_words+1

for i, topic_dist in enumerate(topic_distributions):
    #print topic_dist
    topic_words = np.array(ngrams)[np.argsort(topic_dist)][:-n_top_words:-1]
    importances = np.array(topic_dist)[np.argsort(topic_dist)][:-n_top_words:-1]
    importances = '|'.join(map(str, importances))
    topic_words = '|'.join(topic_words)
    topic_words = topic_words.replace('~','')
    coef = eta_pred[i]
    file_topics.write('Topic ' + repr(i) + '|'+repr(coef)+'|'+topic_words+'|'+importances+'\n')

file_topics.close()

###  topic loadings( only for test data set)

thetas_slda=slda.transform(X)
topic_loadings=thetas_slda.T

import pandas as pd
df = pd.DataFrame(topic_loadings)
df.to_csv(DIRECTORY +"topic_loadings_train.csv")


### get mean and median of topic loadings( only for traing data set)
import statistics

file_results = open(DIRECTORY + 'loading_avg_median_train.txt', 'w')

for i in range(K):
    mean1=statistics.mean(topic_loadings[i])
    median1=statistics.median(topic_loadings[i])
    file_results.write('topic'+repr(i)+'|'+repr(mean1)+repr(median1) + '\n')
file_results.close()



###  topic loadings( only for test data set)

thetas_test_slda = slda.transform(X_test)
topic_loadings_test=thetas_test_slda.T


df = pd.DataFrame(topic_loadings_test)
df.to_csv(DIRECTORY +"topic_loadings_test.csv")


####get mean and median of topic loadings( only for test data set)
file_results = open(DIRECTORY + 'loading_avg_median_test.txt', 'w')

for i in range(K):
    mean2=statistics.mean(topic_loadings_test[i])
    median2=statistics.median(topic_loadings_test[i])
    file_results.write('topic'+repr(i)+'|'+repr(mean2)+repr(median2) + '\n')
file_results.close()

#### OBTAIN OUT-OF-SAMPLE PREDICTIONS FOR Y ####

thetas_test_slda = slda.transform(X_test)
y_slda = [np.dot(eta_pred, thetas_test_slda[i]) for i in range(len(filenames_test))]

file_results = open(DIRECTORY + 'pyoutput_adv.txt', 'w')

for fn,filename in enumerate(filenames_test):
    filename = filename.split('|')[0].strip()
    actual = y_test[fn]
    predicted = y_slda[fn]
    file_results.write(filename+'|'+repr(actual)+'|'+repr(predicted) + '|' + repr(abs(predicted)) + '\n')
file_results.close()

##calculate Predictive R square 
#from sklean.metrics import r2_score
#r2_score(y_test,y_slda)

from statistics import mean
import numpy as np
y_slda2 = np.array(y_slda, dtype=np.float64)
y_test2 = np.array(y_test, dtype=np.float64)
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
r_squared = coefficient_of_determination(y_test2,y_slda2)
print(r_squared)





