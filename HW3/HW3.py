import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
	
def form_dataset(dataframe_string, row_num = None):
	'''
	Function reads the .csv file into a panda dataframe.
	Inputs:
	dataframe_string: String for file name
	filetype: string for file_type
	row_num: integer, number of rows to read if excel or csv
	Outputs:
	Credit_df = A panda dataframe
	'''
	if filetype = 'string':
		df = pd.read_csv(dataframe_string, nrows = row_num)
	elif filetype = 'excel':
		df = pd.read_excel(dataframe_string, nrows = row_num)
	elif filetype = 'json':
		df = pd.read_json(dataframe_string)
	
	return df

def exploration(df):
	'''
	Data exploration function - used to derive information about
	the dataframe for analysis.
	We seperate this information into two main categories,
	Summary_information, which is a description of the dataframe
	Column_Names, which is a list of column headers in the dataframe
	These are stored in a dictionary.
	Input:
	Credit_df: A panda dataframe
	Output:
	Data_dictionary: A dictionary containing two entries, Column_Names and Summary_information
	'''
	summary_information = df.describe()
	header_list = list(df)
	data_dictionary = {}
	data_dictionary["Summary Information"] = summary_information
	data_dictionary["Column Names"] = header_list
	
	return data_dictionary

	
def processing_drop(df, drop_list, maximum):
	'''
	1) Drops all rows where the 'DebtRatio' value is greater than 1
	It is a percentage, so these values are anomalous.
	Input:
	Credit_df: A panda dataframe
	drop_list: List of columns to act on
	maximum: The integer to drop if the value is greater than
	Outputs:
	Credit_df
	'''
	for variable in drop_list:
		df = df[df[variable] <= maximum]
	
	return df

def processing_mean(df, list_to_mean, operation_type, value = None):
	'''
	Fill in null values with the mean of the column.
	Input:
	Credit_df: A panda dataframe
	list_to_mean: List of columns to act on
	Outputs:
	Credit_df
	'''
	
	for variable in list_to_mean:
		if operation_type == 'mean'
			df[variable].fillna(df[variable].mean(), inplace=True)
		elif operation_type == 'median':
			df[variable].fillna(df[variable].median(), inplace=True)
		elif operation_type == 'set':
			df[variable].fillna(value, inplace=True)
	
	return df

def processing_mult(df, multiply_list, multiplier):
	'''
	Multiply all values in column by a multiplier
	Input:
	Credit_df: A panda dataframe
	multiply_list: List of columns to act on
	multiplier: The number to multiply each value by
	Outputs:
	Credit_df
	'''	
	for variable in multiply_list:
		df[variable] = df[variable].apply(lambda x: x * multiplier)
			
	return df


def create_graph(df, variable, column_title, y_label):
	'''
	Take a variable and create a line chart mapping that variable
	against a dependent_variable, serious delinquency in the prior two years
	Inputs:
	credit_df: A panda dataframe
	variable: A string, which is a column in credit_df
	Outputs:
	Variable_chart: A matplotlib object of the resultant chart
	'''
	chart_size = (10, 5)
	columns = [variable, column_title]
	mean_variable = df[columns].groupby(variable).mean()
	variable_chart = mean_variable.plot(kind = 'line',figsize = chart_size)
	
	plt.ylabel(ylabel)
	plt.show()
	
	return variable_chart

def bin_gen(credit_df, variable, label, fix_value):
	'''
	Create a bin column for a given variable, derived by using the 
	description of the column to determine the min, 25, 50, 75 and max
	of the column. Then categorize each value in the original variable's
	column in the new column, labeled binned_<variable>, with 1,2,3,4
	Ranging from min to max
	Inputs:
	df: A panda dataframe
	variable: A string, which is a column in credit_df
	label: A string
	fix_value: Either prefix or suffix
	Outputs:
	df: A panda dataframe
	'''
	variable_min = df[variable].min()
	variable_25 = df[variable].quantile(q = 0.25)
	variable_50 = df[variable].quantile(q = 0.50)
	variable_75 = df[variable].quantile(q = 0.75)
	variable_max = df[variable].max()
	
	bin = [variable_min, variable_25, variable_50, variable_75, variable_max]
	
	if fix_value = 'prefix':
		bin_label = label + variable
	elif fix_value = 'suffix':
		bin_label = variable + label
	
	df[bin_label] = pd.cut(df[variable], bins = bin, include_lowest = True, labels = [1, 2, 3, 4])
	
	return df
	
def dummy_variable(binned_list, df):
	'''
	Using the binned columns, replace them with dummy columns.
	Inputs:
	credit_df: A panda dataframe
	binned_list: A list of column headings for binned variables
	Outputs:
	credit_df:A panda dataframe
	'''
	df = pd.get_dummies(df, columns = binned_list)
	
	return df

def classifier(x, y, threshold, classifier,):
	'''
	Takes the dataframe and runs a decision tree regression on it,
	returning the score for the testing and training sets.
	Removes the variable and binned_variable from the variables being used
	as y.

	Inputs:
	credit_df: A panda dataframe
	variable: A string, which is a column in credit_df
	type_of_df: A string, either binned or dummy, describing which columns in the df are being used.
	Outputs:
	x,y test score and x,y train score
	'''


	
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 32)
	
	tree_model = tree.DecisionTreeRegressor()
	tree_model = tree_model.fit(x_train, y_train)
	predicted_y = tree_model.predict(x_test)

	plt.scatter(y_test, predicted_y)
	plt.xlabel("True "+ variable)
	plt.ylabel("Predicted "+ variable)
	plt.axis()
	plt.show()
	
	threshold = 0.5
	print("Accuracy Score:", accuracy_score(y_test, predicted_y > threshold))
	
	print("Confusion Matrix:", confusion_matrix(y_test, predicted_y > threshold))
	
	return(tree_model.score(x_test, y_test), tree_model.score(x_train, y_train), predicted_y)
	 
	
def execute_homework():
	df = form_dataset("credit-data.csv")
	data_dictionary = exploration(df)
	
	print(data_dictionary["Summary Information"])
	print(data_dictionary["Column Names"])
	
	chart_list = ['NumberOfOpenCreditLinesAndLoans', 'NumberOfDependents', 'age', 'MonthlyIncome']
	#for chart in chart_list:
	#	graph = create_graph(credit_df, chart)

	drop_list = ['DebtRatio']
	credit_df = processing_drop(df, drop_list, 1)

	print(df[df['DebtRatio'] > 1])
	
	list_to_mean = ['MonthlyIncome', 'NumberOfDependents']
	df = processing_mean(df, list_to_mean)
	
	multiply_list = ['DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']
	df = processing_mult(df, multiply_list, 100)
	
	print(df.isna().any())

	
	categorical_list = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome', 'age']
	binned_list = []
	for variable in categorical_list:
		df = bin_gen(credit_df, variable)
		bin_label = "Binned_" + variable
		binned_list.append(bin_label)
	
	print(list(df))
	#credit_df = dummy_variable(binned_list, credit_df)
	
	classify_list = ['age', 'DebtRatio', 'MonthlyIncome']
	for classify_variable in classify_list:
		classify_test, classify_train, classify_hat = classifier(credit_df, classify_variable, 'binned')
		print(classify_test, classify_train, classify_variable)
	
	credit_df = dummy_variable(binned_list, df)
	
	
	
	
	
	
	
