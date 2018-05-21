import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

	
def form_dataset(dataframe_string, row_num = None):
	'''
	Function reads the .csv file into a panda dataframe.
	Inputs:
	dataframe_string: String for file name
	filetype: string for file_type
	row_num: integer, number of rows to read if excel or csv
	Outputs:
	df = A panda dataframe
	'''
	if '.csv' in dataframe_string:
		df = pd.read_csv(dataframe_string, nrows = row_num)
	elif '.xls' in dataframe_string:
		df = pd.read_excel(dataframe_string, nrows = row_num)
	elif '.json' in dataframe_string:
		df = pd.read_json(dataframe_string)
	
	print("Loaded" + dataframe_string)
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
	df: A panda dataframe
	Output:
	Data_dictionary: A dictionary containing two entries, Column_Names and Summary_information
	'''
	summary_information = df.describe()
	header_list = list(df)
	data_dictionary = {}
	data_dictionary["Summary Information"] = summary_information
	data_dictionary["Column Names"] = header_list
	
	return data_dictionary

	
def processing_drop(df, drop_list, target_quantifier, value):
	'''
	1) Drops all rows where the variables in the drop_list value where the target is less than, greater than, or equal to a value.
	Input:
	df: A panda dataframe
	drop_list: List of columns to act on
	maximum: The integer to drop if the value is greater than
	Outputs:
	df
	'''
	for variable in drop_list:
		if target_quantifier == 'equal':
			df = df[df[variable] == value]
		elif target_quantifier == 'greater':
			df = df[df[variable] >= value]
		elif target_quantifier == 'lesser':
			df = df[df[variable] <= value]
	
	return df

def processing_mean(df, list_to_mean, operation_type, value = None):
	'''
	Fill in null values with the mean, median or a set value of the column.
	Input:
	df: A panda dataframe
	list_to_mean: List of columns to act on
	Outputs:
	df
	'''
	
	for variable in list_to_mean:
		if operation_type == 'mean':
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
	df: A panda dataframe
	multiply_list: List of columns to act on
	multiplier: The number to multiply each value by
	Outputs:
	df
	'''	
	for variable in multiply_list:
		df[variable] = df[variable].apply(lambda x: x * multiplier)
			
	return df

def outlier(df, variable):
	'''
	Locate outliers in given column and eliminate them
	'''
	low_out = df[variable].quantile(0.005)
	high_out = df[variable].quantile(0.995)
	df_changed = df.loc[(df[variable] > low_out) & (df[variable] < high_out)]

	number_removed = df.shape[0] - df_changed.shape[0]
	print("Removed" + str(number_removed) + "outliers from" + variable)

	return df_changed

	
def create_graph(df, variable, subject_variable, type, graph_type):
	'''
	Take a variable and create a line chart mapping that variable
	against a dependent_variable, serious delinquency in the prior two years
	Inputs:
	df: A panda dataframe
	variable: A string, which is a column in df
	Outputs:
	Variable_chart: A matplotlib object of the resultant chart
	'''
	columns = [subject_variable, variable]
	if type == 'mean':
		var_plot = df[columns].groupby(subject_variable).mean()
	elif type == 'total':
		var_plot = df[columns].groupby(subject_variable).sum()
	
	graph = var_plot.plot(kind = graph_type, use_index = False, figsize = (10,5))

	return graph

def bin_gen(df, variable, label, fix_value):
	'''
	Create a bin column for a given variable, derived by using the 
	description of the column to determine the min, 25, 50, 75 and max
	of the column. Then categorize each value in the original variable's
	column in the new column, labeled binned_<variable>, with 1,2,3,4
	Ranging from min to max
	Inputs:
	df: A panda dataframe
	variable: A string, which is a column in df
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
	unique_values = len(set(bin))
	
	label_list = []
	iterator = 0
	for x in range(1, unique_values):
		iterator += 1
		label_list.append(iterator)
	
	if fix_value == 'prefix':
		bin_label = label + variable
	elif fix_value == 'suffix':
		bin_label = variable + label
	
	df[bin_label] = pd.cut(df[variable], bins = bin, include_lowest = True, labels = label_list, duplicates = 'drop')
	df.drop([variable], inplace = True, axis=1)
	
	return df
	
def dummy_variable(variable, df):
	'''
	Using the binned columns, replace them with dummy columns.
	Inputs:
	df: A panda dataframe
	variable: A list of column headings for binned variables
	Outputs:
	df:A panda dataframe
	'''
	dummy_df = pd.get_dummies(df[col]).rename(columns = lambda x: str(variable)+ str(x))
	df = pd.concat([df, dummy_df], axis=1)
	df.drop([variable], inplace = True, axis=1)
	
	return df

def print_csv(df, filename):
	df.to_csv(filename)
	print('CSV Created')
	
	
	
	
	
	
