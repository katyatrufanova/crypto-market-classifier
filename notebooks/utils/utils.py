"""
A set of utility functions
"""
import os
from shutil import which, rmtree
import zipfile
import time
import csv
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def make_dataset(source_path, dest_path):
	"""Import the dataset from a remote source and extract the data.
	NOTE: 'WGET' module must be installed on your local machine!
	Args:
		source_path (str): the remote source path.
		dest_path (str): the local destination path.
	Returns:
		None.
	"""
	if which('wget') is not None:
		data_zip = dest_path + 'data.zip'
		if not os.path.isfile(data_zip):
			os.system('wget --no-check-certificate -O '+data_zip+' "'+source_path+'"')
		try:
			for filename in os.listdir(dest_path):
				if filename != '.gitignore' and filename != 'data.zip':
					file_path = os.path.join(dest_path, filename)
					try:
						if os.path.isfile(file_path) or os.path.islink(file_path):
							os.unlink(file_path)
						elif os.path.isdir(file_path):
							rmtree(file_path)
					except Exception as e:
						print('WARNING: failed to delete %s. Reason: %s' % (file_path, e))
			print('Extracting files...')
			with zipfile.ZipFile(data_zip, 'r') as zip_ref:
				zip_ref.extractall(dest_path)
			ko = os.path.join(dest_path, '__MACOSX')
			if os.path.isdir(ko):
				print('Finalizing...')
				rmtree(ko)
				os.unlink(data_zip)
			print('Completed!')
		except OSError as e:
			print(e)
	else:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: missing module! Install\033[95m wget\033[0m on your machine.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def get_raw_data(path, view, exchange, crypto):
	"""Get a csv dataset into a Pandas dataframe.
	Args:
		path (str): the csv folder path.
		view (str): the view name (see global_config.py).
		exchange (str): the exchange name (see global_config.py).
		crypto (str): the crypto name (see global_config.py).
	Returns:
		The dataframe with data.
	"""
	return pd.read_csv(os.path.join(path, view, exchange, crypto+'.csv'), encoding='UTF-8')


def get_train_test_data(
		data,
		columns_to_drop,
		window_size=10,
		min_after=1,
		train_split_date='2023-09-01 00:00:00',
		verbose=False
	):
	"""Split the dataset into training and testing sets.
	Args:
		data (Dataframe): the Pandas Dataframe with data.
		window_size (int): tha size of the sliding sequence.
		min_after (str): time instant after the window to be predicted.
		train_split_date (date): the date until to which samples are considered training data.
		verbose (bool): whether to output or not.
	Returns:
		The splitting of training and testing data and labels.
	"""
	data = data.sort_values(by=['origin_time'])
	limit = len(data[data['origin_time'] < train_split_date])
	features = data.drop(columns=columns_to_drop, axis=1).values
	labels = data['labels'].values
	if verbose:
		print(f'{"Shape of features set: ":<30}{str(features.shape):<15}')
		print(f'{"Shape of labels set: ":<30}{str(labels.shape):<15}\n')

	X, Y = [], []
	for i in range(len(data) - window_size - min_after - 1):
		x = features[i : i + window_size]
		y = labels[i + window_size + min_after]
		X.append(x), Y.append(y)
	X, Y = np.array(X), np.array(Y)
	if verbose:
		print(f'{"Shape of features by window: ":<30}{str(X.shape):<15}')
		print(f'{"Shape of labels by window: ":<30}{str(Y.shape):<15}\n')

	X_train, y_train = X[:limit], Y[:limit]
	X_test, y_test = X[limit:], Y[limit:]
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
	if verbose:
		print(f'{"Shape of training data: ":<30}{str(X_train.shape):<15}{X_train.shape[0] * 100 / len(data):>10.2f}%')
		print(f'{"Shape of training labels: ":<30}{str(y_train.shape):<15}{y_train.shape[0] * 100 / len(data):>10.2f}%')
		print(f'{"Shape of testing data: ":<30}{str(X_test.shape):<15}{X_test.shape[0] * 100 / len(data):>10.2f}%')
		print(f'{"Shape of testing labels: ":<30}{str(y_test.shape):<15}{y_test.shape[0] * 100 / len(data):>10.2f}%')

	return X_train, y_train, X_test, y_test


def train_test_model(X_train, y_train, X_test, y_test, config=False):
	"""Execute the model and compute the metrics for each class.
	Args:
		X_train (array): the training data.
		y_train (array): the training labels.
		X_test (array): the testing data.
		y_test (array): the testing labels.
	Returns:
		The dictionary of metrics computed for each class.
	"""
	warnings.filterwarnings('ignore')

	start_time = time.time()
	if config:
		base_classifier = BaggingClassifier(
			estimator=HistGradientBoostingClassifier(
				learning_rate=config[0],
				max_iter=config[1],
				max_leaf_nodes=config[2],
				random_state=3
			),
			n_estimators=5,
			random_state=3,
			n_jobs=-1
		)
	else:
		base_classifier = BaggingClassifier(
			estimator=HistGradientBoostingClassifier(
				random_state=3
			),
			n_estimators=5,
			random_state=3,
			n_jobs=-1
		)
	base_classifier.fit(X_train, y_train)
	y_pred = base_classifier.predict(X_test)
	end_time = time.time()

	matrix = confusion_matrix(y_test, y_pred)
	a = matrix.diagonal() / matrix.sum(axis=1)
	p = precision_score(y_test, y_pred, labels=base_classifier.classes_, average=None)
	r = recall_score(y_test, y_pred, labels=base_classifier.classes_, average=None)
	f = f1_score(y_test, y_pred, labels=base_classifier.classes_, average=None)
	t = end_time - start_time
	a, p, r, f = list(a), list(p), list(r), list(f)
	metrics = {
		'accuracy_down': a[0],
		'accuracy_neutral': a[1],
		'accuracy_up': a[2],
		'precision_down': p[0],
		'precision_neutral': p[1],
		'precision_up': p[2],
		'recall_down': r[0],
		'recall_neutral': r[1],
		'recall_up': r[2],
		'f1_down': f[0],
		'f1_neutral': f[1],
		'f1_up': f[2],
		'exec_time': t
	}
	if config:
		metrics['learning_rate'] = config[0]
		metrics['max_iter'] = config[1]
		metrics['max_leaf_nodes'] = config[2]
	return metrics


def save_results(file, results, view, crypto):
	"""Save the experiments results to file.
	Args:
		file (str): the file path where to save data.
		results (dict): the metrics of the experiment.
		view (str): the data view of the experiment.
		crypto (array): the crypto of the experiment.
	Returns:
		None.
	"""
	alpha = list(map(chr, range(97, 123)))
	if os.path.isfile(file):
		with open(file, 'r', encoding='utf-8') as outfile:
			csvreader = csv.reader(outfile, delimiter=',', quotechar='"')
			view_ids = {}
			for row in csvreader:
				view_ids[row[1]] = ''
			del view_ids['view']
			if view in view_ids.keys():
				index = [i for i, v in enumerate(view_ids.keys()) if v == view][0]
			else:
				view_ids[view] = ''
				index = len(view_ids.keys()) - 1
		with open(file, 'a', encoding='utf-8') as outfile:
			base_dict = {'view_id': alpha[index].upper(), 'view': view, 'crypto': crypto}
			my_dict = {**base_dict, **results}
			csvwriter = csv.writer(outfile, delimiter=',')
			csvwriter.writerow(my_dict.values())
	else:
		with open(file, 'w', encoding='utf-8') as outfile:
			base_dict = {'view_id': alpha[0].upper(), 'view':view, 'crypto': crypto}
			my_dict = {**base_dict, **results}
			csvwriter = csv.writer(outfile, delimiter=',')
			csvwriter.writerow(my_dict)
			csvwriter.writerow(my_dict.values())
