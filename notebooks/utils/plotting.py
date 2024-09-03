"""
A set of plotting functions
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(grouped, x_title):
	"""Plot the metrics of the result file.
	Args:
		grouped (dataframe): a grouped version of a dataframe.
		x_title (str): values label on X axis
	Returns:
		None.
	"""
	data = {
		'accuracy': {
			'down': grouped['accuracy_down'].values,
			'neutral': grouped['accuracy_neutral'].values,
			'up': grouped['accuracy_up'].values
		},
		'precision': {
			'down': grouped['precision_down'].values,
			'neutral': grouped['precision_neutral'].values,
			'up': grouped['precision_up'].values
		},
		'recall': {
			'down': grouped['recall_down'].values,
			'neutral': grouped['recall_neutral'].values,
			'up': grouped['recall_up'].values
		},
		'f1': {
			'down': grouped['f1_down'].values,
			'neutral': grouped['f1_neutral'].values,
			'up': grouped['f1_up'].values
		},
	}
	counter = 0
	titles = list(data.keys())
	fig, axs = plt.subplots(2, 2, figsize=(18, 12))
	for _, axl in enumerate(axs):
		for _, ax in enumerate(axl):
			ax.plot(data[titles[counter]]['down'], 'o-', label='down')
			ax.plot(data[titles[counter]]['neutral'], 'o-', label='neutral')
			ax.plot(data[titles[counter]]['up'], 'o-', label='up')
			ax.set_title('Avg. ' + titles[counter].capitalize() + ' by ' + x_title, fontsize=18)
			ax.set_xticks(np.arange(len(list(grouped.index))), list(grouped.index))
			ax.legend(loc = 'upper right', ncol = 1)
			counter = counter + 1
	fig.tight_layout()
	plt.show()


def plot_gain(g1, g2):
	"""Plot the gain between two set of observations.
	Args:
		g1 (dataframe): a grouped version of the default results.
		g2 (dataframe): a grouped version of the hyperparameters tuning results.
	Returns:
		None.
	"""
	data_default = pd.DataFrame({
		'accuracy': [i / 3 for i in list(map(sum, zip(g1['accuracy_down'], g1['accuracy_neutral'], g1['accuracy_up'])))],
		'precision': [i / 3 for i in list(map(sum, zip(g1['precision_down'], g1['precision_neutral'], g1['precision_up'])))],
		'recall': [i / 3 for i in list(map(sum, zip(g1['recall_down'], g1['recall_neutral'], g1['recall_up'])))],
		'f1': [i / 3 for i in list(map(sum, zip(g1['f1_down'], g1['f1_neutral'], g1['f1_up'])))]
	})
	data_tuning = pd.DataFrame({
		'accuracy': [i / 3 for i in list(map(sum, zip(g2['accuracy_down'], g2['accuracy_neutral'], g2['accuracy_up'])))],
		'precision': [i / 3 for i in list(map(sum, zip(g2['precision_down'], g2['precision_neutral'], g2['precision_up'])))],
		'recall': [i / 3 for i in list(map(sum, zip(g2['recall_down'], g2['recall_neutral'], g2['recall_up'])))],
		'f1': [i / 3 for i in list(map(sum, zip(g2['f1_down'], g2['f1_neutral'], g2['f1_up'])))]
	})
	counter = 0
	titles = list(data_default.keys())
	fig, axs = plt.subplots(2, 2, figsize=(18, 12))
	for _, axl in enumerate(axs):
		for _, ax in enumerate(axl):
			ax.plot(data_default[titles[counter]].values, 'o-', label='default_param')
			ax.plot(data_tuning[titles[counter]].values, 'o-', label='best_param')
			ax.set_title('Avg. ' + titles[counter].capitalize() + ' Gain', fontsize=18)
			ax.set_xticks(np.arange(len(list(g1.index))), list(g1.index))
			ax.legend(loc = 'upper right', ncol = 1)
			counter = counter + 1
	fig.tight_layout()
	plt.show()


def plot_tradeoff(grouped):
	"""Plot the trade-off between metrics and execution times.
	Args:
		grouped (dataframe): a grouped version of a dataframe.
	Returns:
		None.
	"""
	data = pd.DataFrame({
		'accuracy': [i / 3 for i in list(map(sum, zip(grouped['accuracy_down'], grouped['accuracy_neutral'], grouped['accuracy_up'])))],
		'precision': [i / 3 for i in list(map(sum, zip(grouped['precision_down'], grouped['precision_neutral'], grouped['precision_up'])))],
		'recall': [i / 3 for i in list(map(sum, zip(grouped['recall_down'], grouped['recall_neutral'], grouped['recall_up'])))],
		'f1': [i / 3 for i in list(map(sum, zip(grouped['f1_down'], grouped['f1_neutral'], grouped['f1_up'])))],
		'exec_time': grouped['exec_time']
	})
	counter = 0
	titles = list(data.keys())
	fig, axs = plt.subplots(2, 2, figsize=(18, 12))
	for _, axl in enumerate(axs):
		for _, ax in enumerate(axl):
			ax.scatter(data[titles[len(titles) - 1]].values, data[titles[counter]].values, s=80)
			ax.set_title(titles[counter].capitalize()+' by '+titles[len(titles) - 1].capitalize(), fontsize=18)
			ax.set_xlabel(titles[len(titles) - 1].capitalize() + ' (sec, log base2)', fontsize=14)
			ax.grid()
			for i, (xi, yi) in enumerate(zip(data[titles[len(titles) - 1]].values, data[titles[counter]].values)):
				ax.text(xi+.001, yi, list(grouped.index)[i], fontsize=14)
			ax.set_xscale('log', base=2)
			counter = counter + 1
	fig.tight_layout()
	plt.show()


def print_best_params(results_default, results_tuning, granularity='view'):
	"""Plot the trade-off between metrics and execution times.
	Args:
		results_default (dataframe): the default results.
		results_tuning (dataframe): the hyperparameters tuning results.
		granularity (str): how to group results. 'view' or 'crypto' allowed.
	Returns:
		None.
	"""
	grouped = results_tuning.drop(columns=['view'])
	grouped = grouped.groupby(['view_id', 'crypto']).max()
	metrics = ['accuracy_down', 'accuracy_neutral', 'accuracy_up', 'precision_down', 'precision_neutral', 'precision_up', 'recall_down', 'recall_neutral', 'recall_up', 'f1_down', 'f1_neutral', 'f1_up']
	best_config = pd.DataFrame({
		'view_id':[],'crypto':[],'learning_rate':[],'max_iter':[],'max_leaf_nodes':[]
	})
	print(''.join(['> ' for i in range(42)]))
	print(f'\n{"BEST PARAMS":>52}')
	print(f'\n{"ID":<8}{"CRYPTO":<10}{"learning_rate":>18}{"max_iter":>18}{"max_leaf_nodes":>18}\n')
	print(''.join(['> ' for i in range(42)]))
	for i in grouped.index:
		d = results_default.loc[(results_default['view_id'] == i[0]) & (results_default['crypto'] == i[1])]
		t = results_tuning.loc[(results_tuning['view_id'] == i[0]) & (results_tuning['crypto'] == i[1])]
		max_ids = [t[m].idxmax() for m in metrics]
		best_row = results_tuning.iloc[max(set(max_ids), key=max_ids.count)]
		score = [1 for m in metrics if d[m].values > best_row[m]]
		if sum(score) > (len(metrics) / 2):
			best_config.loc[len(best_config)] = [i[0], i[1], .01, 100, 31]
		else:
			best_config.loc[len(best_config)] = [
				i[0], i[1], best_row["learning_rate"], int(best_row["max_iter"]), int(best_row["max_leaf_nodes"])
			]
	if granularity == 'view':
		lr = best_config.groupby(['view_id'])['learning_rate'].apply(pd.Series.mode)
		mi = best_config.groupby(['view_id'])['max_iter'].apply(pd.Series.mode)
		mln = best_config.groupby(['view_id'])['max_leaf_nodes'].apply(pd.Series.mode)
		ids = list(set(best_config['view_id'].values))
		for i, v in enumerate(sorted(ids)):
			print(f'{v:<8}{"avg.":<10}{lr[i]:>18}{mi[i]:>18}{mln[i]:>18}')
	elif granularity == 'crypto':
		for _, v in best_config.iterrows():
			print(f'{v["view_id"]:<8}{v["crypto"]:<10}{v["learning_rate"]:>18}{v["max_iter"]:>18}{v["max_leaf_nodes"]:>18}')