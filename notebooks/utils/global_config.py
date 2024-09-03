"""
The collection of global configurations
"""
CONFIG = {
	'VIEW':					[
								'candles',
								'snapshot'
							],
	'EXCHANGE':				[
								'BINANCE',
								'HOUBI',
								'OKX'
							],
	'CRYPTO':				[
								'AVAX',
								'BNB',
								'DOGE',
								'DOT',
								'NEO',
								'SOL'
							],
	'REMOTE_DATA':			'<READACTED>',
	'DATA_PATH':			'data/',
	'RESULT_FILE_DEFAULT':	'reports/results_default.csv',
	'RESULT_FILE_TUNING':	'reports/results_tuning.csv',
	'FEATURES_EXCLUDED':	{
								'candles': ['origin_time', 'start', 'stop', 'exchange', 'symbol', 'labels'],
								'snapshot': ['origin_time', 'exchange', 'symbol', 'labels']
							}
}


def get_global_config():
	"""
	The getter method
	"""
	return CONFIG


def set_global_config(key, value):
	"""
	The setter method
	"""
	CONFIG.update({key: value})