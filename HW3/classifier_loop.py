#Code Based on Rayid Ghani's Magic Loop: https://github.com/rayidghani/magicloops/blob/master/magicloop.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import time
import seaborn as sns
import random
import pylab as pl

def establish_grid():

grid = {'Forest':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'Tree': {'criterion': ['gini', 'entropy'], 'max_depth': [5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
    'Bagging':{'n_estimators    ':[1,10,20,50], 'max_samples':[5,10], 'max_features': [5,10]},
    'KNN' :{'n_neighbors': [1,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
    'Boosted': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
    'Logit': {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10]},
    'SVM' :{'C' :[0.01,0.1,1,10],'kernel':['linear']},
    }

def establish_classifiers():

	classifiers = {'Forest': RandomForestClassifier(),
		'Tree': DecisionTreeClassifier(),
		'Bagging': BaggingClassifier(),
		'KNN': KNeighborsClassifier() 
		'Boosted': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
		'Logit': LogisticRegression(),
		'SVM': SVC(probability=True, random_state=0),
		}


def scores_at_k(y_true, y_scores, k):
    '''
    For a given level of k, calculate corresponding
    precision, recall, and f1 scores.
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1

def clf_loop(x, y, models):
    '''
    '''
	iterator = 0
    
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state = 32)
	
    for index , classifier in enumerate([classifiers[x] for x in models]):
		parameters = grid[models[index]]
        for x in ParameterGrid(parameter_values):
            try:
                start = time.time()
				iterator += 1
                classifier.set_params(** x)
				accuracy = classifier.score(x_test, y_test)
                y_hat = classifier.fit(x_train, y_train).predict_proba(x_test)[:,1]

                end = time.time()
				duration = end - start

				results_df = pd.DataFrame(columns=('model_type','parameters', 'duration', 'accuracy','Average Precision Score'))
                results_df.loc[iterator] = [models[index], x, duration, accuracy, average_precision_score(y_test, y_hat)]
				
				create_precision_recall_graph(y_true,y_hat, models[index], x)

            except IndexError:
                continue
				
    return results_df
	

def create_precision_recall_graph(y_true,y_hat, model, parameter):
    '''
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)   
	
    plt.clf()
    plt.plot(recall, precision, color='blue', label = 'Precision Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title('Graph of Precision Recall Curve for {} on {}'.format(model, parameter))
    plt.legend()
    
	plt.show()
