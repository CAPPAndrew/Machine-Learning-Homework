#Code Based on Rayid Ghani's Magic Loop: https://github.com/rayidghani/magicloops/blob/master/simpleloop.py

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
	grid = {'Forest':{'n_estimators': [10,100], 'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
	'Tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
	'KNN' :{'n_neighbors': [1,50],'weights': ['uniform','distance'],'algorithm': ['auto']},
	'Boosted' : {'algorithm': ['SAMME'], 'n_estimators': [1,10,100,1000]},
	'Logit': {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10]},
	'SVM' :{'C' :[0.1,1],'kernel':['linear']}
	}
	
	return grid
def establish_classifiers():
	classifiers = {'Forest': RandomForestClassifier(),
		'Tree': DecisionTreeClassifier(),
		'KNN': KNeighborsClassifier(),
		'Boosted': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
		'Logit': LogisticRegression(),
		'SVM': SVC(probability=True, random_state=0),
		}
	
	return classifiers
	
def clf_loop(x, y, models, grid, classifiers):
	'''
	Perform classifiers on given x and y inputs.
	Inputs:
	x = variable
	y = Subject variables
	models = list of models to use
	'''
	if models == 'all':
		models = ['Forest', 'Tree', 'KNN', 'Boosted', 'Logit', 'SVM']
	iterator = -1
    
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 12)
	
	results_df = pd.DataFrame(columns=('model_type','parameters', 'duration', 'accuracy','Average Precision Score', 'AUC ROC Score'))

	
	for index , classifier in enumerate([classifiers[x] for x in models]):
		parameters = grid[models[index]]
		for x in ParameterGrid(parameters):
			try:
				start = time.time()
				iterator += 1
				classifier.set_params(** x)

				y_hat = classifier.fit(x_train, y_train.values.ravel()).predict_proba(x_test)[:,1]
				
				end = time.time()
				
				accuracy = classifier.score(x_test, y_test)
				duration = end - start
				
				results_df.loc[iterator] = [models[index], x, duration, accuracy, average_precision_score(y_test, y_hat),
				roc_auc_score(y_test, y_hat)]

				create_precision_recall_graph(y_test,y_hat, models[index], x)

			except IndexError:
				continue
				
	return results_df
	

def create_precision_recall_graph(y_true, y_hat, model, parameter):
	'''
	Create a precision recall graph based on given values and label it based on the model and parameters.
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
